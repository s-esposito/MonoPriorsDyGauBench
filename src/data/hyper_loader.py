import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import cv2 as cv
import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
from typing import NamedTuple
from torch.utils.data import Dataset
import copy
from typing import Optional
import re

from .base import CameraInfo
from src.data.utils import Camera
from src.utils.general_utils import PILtoTorch
# from scene.dataset_readers import 
from src.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


def extract_prefix_and_id(image_name):
    # only contain direct file name
    image_name = image_name.split('/')[-1]
    
    # Check for the format "name_id.extension"
    match = re.match(r'(.*?)_(\d+)\.(png|jpg)', image_name)
    if match:
        prefix = match.group(1)
        image_id = int(match.group(2))
        return prefix, image_id
    
    # Check for the format "id_name.extension"
    match = re.match(r'(\d+)_(.*?)\.(png|jpg)', image_name)
    if match:
        image_id = int(match.group(1))
        prefix = match.group(2)
        return prefix, image_id
    
    # If neither format matches, try to extract the integer ID from the beginning of the name
    match = re.search(r'^(\d+)', image_name)
    if match:
        image_id = int(match.group(1))
        return None, image_id
    
    # If no ID is found, return the entire name as the prefix and None as the ID
    return os.path.splitext(image_name)[0], None

class Load_hyper_data(Dataset):
    def __init__(self, 
                 datadir, 
                 ratio=1.0,
                 use_bg_points=False,
                 split="train",
                 eval=False,
                 load_flow=False,
                 ):
        
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split
        if eval:
            if len(self.val_id) == 0:
                self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                                (i%4 == 0)])
                self.i_test = self.i_train+2
                self.i_test = self.i_test[:-1,]
            else:
                self.train_id = dataset_json['train_ids']
                self.i_test = []
                self.i_train = []
                for i in range(len(self.all_img)):
                    id = self.all_img[i]
                    if id in self.val_id:
                        self.i_test.append(i)
                    if id in self.train_id:
                        self.i_train.append(i)
        else:
            self.i_train = np.array([i for i in np.arange(len(self.all_img))])
            self.i_test = self.i_train+0
        

        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        self.i_video = [i for i in range(len(self.all_img))]
        self.i_video.sort()
        # all poses
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            camera = camera.scale(ratio)
            camera.position -=  self.scene_center
            camera.position *=  self.coord_scale
            self.all_cam_params.append(camera)

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]
        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}
        self.image_one = Image.open(self.all_img[0])
        #assert False, self.image_one
        self.image_one_torch = PILtoTorch(self.image_one,None).to(torch.float32)

        self.load_flow = load_flow
        if self.load_flow:
            
            # Create dictionaries to store the mapping of (prefix, image_id) to index
            all_dict = {extract_prefix_and_id(self.all_img[idx]): idx for idx in range(len(self.all_img))}

            # Initialize the lists
            self.i_train_prev = [None] * len(self.i_train)
            self.i_train_post = [None] * len(self.i_train)
            self.i_test_prev = [None] * len(self.i_test)
            self.i_test_post = [None] * len(self.i_test)

            # Populate self.i_train_prev and self.i_train_post
            for idx, train_idx in enumerate(self.i_train):
                prefix, image_id = extract_prefix_and_id(self.all_img[train_idx])
                if (prefix, image_id - 1) in all_dict:
                    self.i_train_prev[idx] = all_dict[(prefix, image_id - 1)]
                if (prefix, image_id + 1) in all_dict:
                    self.i_train_post[idx] = all_dict[(prefix, image_id + 1)]

            # Populate self.i_test_prev and self.i_test_post
            for idx, test_idx in enumerate(self.i_test):
                prefix, image_id = extract_prefix_and_id(self.all_img[test_idx])
                if (prefix, image_id - 1) in all_dict:
                    self.i_test_prev[idx] = all_dict[(prefix, image_id - 1)]
                if (prefix, image_id + 1) in all_dict:
                    self.i_test_post[idx] = all_dict[(prefix, image_id + 1)]
                    #    # strictly sort ids by time
            
            '''
            for idx_prev, idx, idx_post in zip(self.i_train_prev, self.i_train, self.i_train_post):
                if idx_prev is not None and idx_post is not None:
                    print(self.all_img[idx_prev].split('/')[-1], 
                    self.all_img[idx].split('/')[-1], 
                    self.all_img[idx_post].split('/')[-1])
                elif idx_prev is None:
                    print(None, self.all_img[idx].split('/')[-1], 
                    self.all_img[idx_post].split('/')[-1])
                else:
                    print(self.all_img[idx_prev].split('/')[-1], 
                    self.all_img[idx].split('/')[-1], 
                    None)
            
            for idx_prev, idx, idx_post in zip(self.i_test_prev, self.i_test, self.i_test_post):
                if idx_prev is not None and idx_post is not None:
                    print(self.all_img[idx_prev].split('/')[-1], 
                    self.all_img[idx].split('/')[-1], 
                    self.all_img[idx_post].split('/')[-1])
                elif idx_prev is None:
                    print(None, self.all_img[idx].split('/')[-1], 
                    self.all_img[idx_post].split('/')[-1])
                else:
                    print(self.all_img[idx_prev].split('/')[-1], 
                    self.all_img[idx].split('/')[-1], 
                    None)
            
            assert False
            '''
        #    self.i_train = sorted(self.i_train, key=lambda x: self.all_time[x])
        #    self.i_test = sorted(self.i_test, key=lambda x: self.all_time[x])
        #    self.first_ids = [idx for idx in range(len(self.all_time)) if (self.all_time[idx] == self.min_rand_xyz)]
        
    def __getitem__(self, index):
        if self.load_flow:
            if self.split == "train":
                return self.load_raw_flow(self.i_train[index],
                    self.i_train_prev[index], self.i_train_post[index])
            elif self.split == "test":
                return self.load_raw_flow(self.i_test[index],
                    self.i_test_prev[index], self.i_test_post[index])
            elif self.split == "video":
                assert False, "Not Implemented Yet"
                return self.load_video_flow(self.i_video[index])
        if self.split == "train":
            return self.load_raw(self.i_train[index])
 
        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        elif self.split == "video":
            return self.load_video(self.i_video[index])
    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            # return len(self.i_video)
            return len(self.video_v2)
    def load_video(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        w = self.image_one.size[0]
        h = self.image_one.size[1]
        # image = PILtoTorch(image,None)
        # image = image.to(torch.float32)
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        try:
            FovY = focal2fov(camera.focal_length[-1], self.h)
            FovX = focal2fov(camera.focal_length[0], self.w)
        except:
            FovY = focal2fov(camera.focal_length, self.h)
            FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=self.image_one_torch,
                              image_path=image_path, image_name=image_name, width=w, height=h, time=time,
                              )
        self.map[idx] = caminfo
        return caminfo  
    def load_raw(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        image = Image.open(self.all_img[idx])
        w = image.size[0]
        h = image.size[1]
        image = PILtoTorch(image,None)
        image = image.to(torch.float32)
        

        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        try:
            FovY = focal2fov(camera.focal_length[-1], h)
            FovX = focal2fov(camera.focal_length[0], w)
        except:
            FovY = focal2fov(camera.focal_length, h)
            FovX = focal2fov(camera.focal_length, w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]

        depth_path = image_path + "_midasdepth"
        depth_name = image_name.split(".")[0]+"-dpt_beit_large_512.png"
        if os.path.exists(os.path.join(depth_path, depth_name)):
            depth = cv.imread(os.path.join(depth_path, depth_name), -1) / (2 ** 16 - 1)
            depth = depth.astype(float)
            depth = torch.from_numpy(depth.copy())
        else:
            depth = None

        '''
        flow_path = image_path + "_flow"
        fwd_flow_path = os.path.join(flow_path, f'{os.path.splitext(image_name)[0]}_fwd.npz')
        bwd_flow_path = os.path.join(flow_path, f'{os.path.splitext(image_name)[0]}_bwd.npz')
        #print(fwd_flow_path, bwd_flow_path)
        #assert False, "Check flow paths"
        if os.path.exists(fwd_flow_path):
            fwd_data = np.load(fwd_flow_path)
            fwd_flow = torch.from_numpy(fwd_data['flow'])
            fwd_flow_mask = torch.from_numpy(fwd_data['mask'])
        else:
            fwd_flow, fwd_flow_mask  = None, None
        if os.path.exists(bwd_flow_path):
            bwd_data = np.load(bwd_flow_path)
            bwd_flow = torch.from_numpy(bwd_data['flow'])
            bwd_flow_mask = torch.from_numpy(bwd_data['mask'])
        else:
            bwd_flow, bwd_flow_mask  = None, None
        '''
        #fwd_flow, fwd_flow_mask, bwd_flow, bwd_flow_mask = None, None, None, None

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=w, height=h, time=time,
                              depth=depth, 
                              )
        self.map[idx] = caminfo
        return caminfo  
    def load_raw_flow(self, idx, idx_prev, idx_post):
        if idx in self.map.keys():
            return self.map[idx]
        
        # read current camera's parameters
        camera = self.all_cam_params[idx]
        image = Image.open(self.all_img[idx])        
        w = image.size[0]
        h = image.size[1]
        image = PILtoTorch(image,None)
        image = image.to(torch.float32)
        

        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        try:
            FovY = focal2fov(camera.focal_length[-1], h)
            FovX = focal2fov(camera.focal_length[0], w)
        except:
            FovY = focal2fov(camera.focal_length, h)
            FovX = focal2fov(camera.focal_length, w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]

        depth_path = image_path + "_midasdepth"
        depth_name = image_name.split(".")[0]+"-dpt_beit_large_512.png"
        if os.path.exists(os.path.join(depth_path, depth_name)):
            depth = cv.imread(os.path.join(depth_path, depth_name), -1) / (2 ** 16 - 1)
            depth = depth.astype(float)
            depth = torch.from_numpy(depth.copy())
        else:
            depth = None

        flow_path = image_path + "_flow"
        
        
        if idx_prev is None:
            R_prev = None
            T_prev = None
            FovY_prev = None
            FovX_prev = None
            time_prev = None
            bwd_flow, bwd_flow_mask = None, None
        else:
            # read previous camera's parameters
            camera_prev = self.all_cam_params[idx_prev]
            
            time_prev = self.all_time[idx_prev]
            R_prev = camera_prev.orientation.T
            T_prev = - camera_prev.position @ R
            try:
                FovY_prev = focal2fov(camera_prev.focal_length[-1], h)
                FovX_prev = focal2fov(camera_prev.focal_length[0], w)
            except:
                FovY_prev = focal2fov(camera_prev.focal_length, h)
                FovX_prev = focal2fov(camera_prev.focal_length, w)

            image_name_prev = self.all_img[idx_prev].split("/")[-1]
            bwd_flow_path = os.path.join(flow_path, f'{os.path.splitext(image_name_prev)[0]}_bwd.npz')
            bwd_data = np.load(bwd_flow_path)
            bwd_flow = torch.from_numpy(bwd_data['flow'])
            bwd_flow_mask = torch.from_numpy(bwd_data['mask'])


        if idx_post is None:
            R_post = None
            T_post = None
            FovY_post = None
            FovX_post = None
            time_post = None
            fwd_flow, fwd_flow_mask = None, None
        else:
            # read post camera's parameters
            camera_post = self.all_cam_params[idx_post]
            time_post = self.all_time[idx_post]
            R_post = camera_post.orientation.T
            T_post = - camera_post.position @ R
            try:
                FovY_post = focal2fov(camera_post.focal_length[-1], h)
                FovX_post = focal2fov(camera_post.focal_length[0], w)
            except:
                FovY_post = focal2fov(camera_post.focal_length, h)
                FovX_post = focal2fov(camera_post.focal_length, w)

            fwd_flow_path = os.path.join(flow_path, f'{os.path.splitext(image_name)[0]}_fwd.npz')
            fwd_data = np.load(fwd_flow_path)
            fwd_flow = torch.from_numpy(fwd_data['flow'])
            fwd_flow_mask = torch.from_numpy(fwd_data['mask'])

        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=w, height=h, time=time,
                              depth=depth, 
                              R_prev=R_prev, T_prev=T_prev, FovY_prev=FovY_prev, FovX_prev=FovX_prev, time_prev=time_prev,
                              R_post=R_post, T_post=T_post, FovY_post=FovY_post, FovX_post=FovX_post, time_post=time_post,
                              fwd_flow=fwd_flow, fwd_flow_mask=fwd_flow_mask,
                              bwd_flow=bwd_flow, bwd_flow_mask=bwd_flow_mask,
                              )

        
        self.map[idx] = caminfo
        return caminfo  

        
def format_hyper_data(data_class, split):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    # dataset = data_class.copy()
    # dataset.mode = split
    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx)):
        camera = data_class.all_cam_params[index]
        # image = Image.open(data_class.all_img[index])
        # image = PILtoTorch(image,None)
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        try:
            FovY = focal2fov(camera.focal_length[-1], data_class.h)
            FovX = focal2fov(camera.focal_length[0], data_class.w)
        except:
            FovY = focal2fov(camera.focal_length, data_class.h)
            FovX = focal2fov(camera.focal_length, data_class.w)
        
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=int(data_class.w), height=int(data_class.h), time=time,
                              )
        cam_infos.append(cam_info)
    return cam_infos
        # matrix = np.linalg.inv(np.array(poses))
        # R = -np.transpose(matrix[:3,:3])
        # R[:,0] = -R[:,0]
        # T = -matrix[:3, 3]