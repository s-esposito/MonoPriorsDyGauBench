import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from src.RAFT.raft import RAFT
from src.RAFT.utils import flow_viz
from src.RAFT.utils.utils import InputPadder

#from flow_utils import *

DEVICE = 'cuda'


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_image(imfile):
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:,:,0] += np.arange(w)
    flow_new[:,:,1] += np.arange(h)[:,np.newaxis]

    res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2

    return fwd_mask, bwd_mask

def run(args, images, input_path, output_path):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        if args.dataset_path.endswith("lego"):
            for item in images:
                assert item.split("/")[-1].startswith("r_")
            images.sort(key=lambda item: int(item.split("/")[-1][2:-4]))
        else:
            images = sorted(images)
        #i = len(images)-2
        #assert False, [len(images), images[0], images[1], images[i], images[i+1]]
        for i in range(len(images) - 1):
            
            image_name = os.path.splitext(os.path.basename(images[i]))[0]
            print(i, image_name)
            image1 = load_image(images[i])
            image2 = load_image(images[i + 1])
            

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_fwd = model(image1, image2, iters=20, test_mode=True)
            _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

            # Save flow
            np.savez(os.path.join(output_path, f'{image_name}_fwd.npz'), flow=flow_fwd, mask=mask_fwd)
            np.savez(os.path.join(output_path, f'{image_name}_bwd.npz'), flow=flow_bwd, mask=mask_bwd)

            # Save flow_img
            #Image.fromarray(flow_viz.flow_to_image(flow_fwd)).save(os.path.join(output_img_path, f'{image_name}_fwd.png'))
            #Image.fromarray(flow_viz.flow_to_image(flow_bwd)).save(os.path.join(output_img_path, f'{image_name}_bwd.png'))

            #Image.fromarray(mask_fwd).save(os.path.join(output_img_path, f'{image_name}_fwd_mask.png'))
            #Image.fromarray(mask_bwd).save(os.path.join(output_img_path, f'{image_name}_bwd_mask.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='Dataset path')
    parser.add_argument("--input_dir", type=str, help='Input image directory')
    parser.add_argument('--model', help="restore RAFT checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    input_path = os.path.join(args.dataset_path, args.input_dir)
    output_path = os.path.join(args.dataset_path, f'{args.input_dir}_flow')
    #output_img_path = os.path.join(args.dataset_path, f'{args.input_dir}_flow_png')
    create_dir(output_path)
    #create_dir(output_img_path)

    


    left_images = glob.glob(os.path.join(input_path, '*left*.png')) + glob.glob(os.path.join(input_path, '*left*.jpg'))
    right_images = glob.glob(os.path.join(input_path, '*right*.png')) + glob.glob(os.path.join(input_path, '*right*.jpg'))
    rest_images = [os.path.join(input_path, img) for img in os.listdir(input_path) if (img.endswith('.png') or img.endswith('.jpg'))]    
    rest_images = [img for img in rest_images if img not in left_images and img not in right_images]
    
    #assert False, rest_images
    #assert False, [left_images[:5], right_images[:5], rest_images[:5]]
    #assert False, input_path#images
    if len(left_images):
        run(args, left_images, input_path, output_path)
    if len(right_images):
        run(args, right_images, input_path, output_path)
    if len(rest_images):
        run(args, rest_images, input_path, output_path)
    
