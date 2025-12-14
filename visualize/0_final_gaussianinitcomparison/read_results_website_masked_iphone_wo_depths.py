import imageio
import numpy as np
import cv2
import os
from tqdm import tqdm
import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt

dataset_mapper = {
    # "dnerf": "DNeRF",
    # "hypernerf": "HyperNeRF",
    "iphone": "iPhone",
    # "nerfds": "NeRF-DS",
    # "nerfies": "Nerfies",
    
}

scenes = {
    # "dnerf": ["bouncingballs", "standup", "trex"],
    # "hypernerf": ["espresso", "torchocolate", "vrig-peel-banana"],
    # "iphone": ["mochi-high-five"] # ["apple", "mochi-high-five", "paper-windmill"],
    "iphone": ["apple"] #, "backpack", "block", "creeper", "handwavy", "haru-sit", "mochi-high-five", "spin", "sriracha-tree", "teddy", "paper-windmill", "pillow"]
    # "nerfds": ["as", "plate", "sieve"],
    # "nerfies": ["tail"], #["broom", "curls", "toby-sit"], #,
    
    # "nerfies": ["curls"],
    # "dnerf": ["standup"],
}

teleport = {
    "bouncingballs": False,
    "standup": False,
    "trex": False,
    "espresso": False,
    "torchocolate": False,
    "vrig-peel-banana": True,
    "apple": False,
    "backpack": False,
    "block": False,
    "creeper": False,
    "handwavy": False,
    "haru-sit": False,
    "mochi-high-five": False,
    "paper-windmill": False,
    "pillow": False,
    "spin": False,
    "sriracha-tree": False,
    "teddy": False,
    "curls": True,
    "broom": True,
    "tail": True,
    "toby-sit": True,
    "as": True,
    "plate": True,
    "sieve": True,
}


method_mapper = {
    "Curve-DepthSupervision-videoda/vanilla": "GT",
    
    "Curve-DepthSupervision-videoda/vanilla": "Depth Supervision (VideoDA)", # <---- it is important that a depth method is the first one here !!!
    "Curve-DepthSupervision-depth-pro/vanilla": "Depth Supervision (Depth Pro)",
    "Curve-DepthSupervision-mega-sam/vanilla": "Depth Supervision (MegaSaM)",
    "Curve-DepthSupervision+GaussianInit-videoda/vanilla": "GaussianInit (VideoDA)", # <---- it is important that a depth method is the first one here !!!
    "Curve-DepthSupervision+GaussianInit-depth-pro/vanilla": "GaussianInit (Depth Pro)",
    "Curve-DepthSupervision+GaussianInit-mega-sam/vanilla": "GaussianInit (MegaSaM)",
    "Curve/vanilla": "EffGS",
    
    "MLP-DepthSupervision-videoda/vanilla": "Depth Supervision (VideoDA)",
    "MLP-DepthSupervision-depth-pro/vanilla": "Depth Supervision (Depth Pro)",
    "MLP-DepthSupervision-mega-sam/vanilla": "Depth Supervision (MegaSaM)",
    "MLP-DepthSupervision+GaussianInit-videoda/vanilla": "GaussianInit (VideoDA)",
    "MLP-DepthSupervision+GaussianInit-depth-pro/vanilla": "GaussianInit (Depth Pro)",
    "MLP-DepthSupervision+GaussianInit-mega-sam/vanilla": "GaussianInit (MegaSaM)",
    "MLP/vanilla": "DeformableGS",
    
    "HexPlane-DepthSupervision-videoda/vanilla": "Depth Supervision (VideoDA)",
    "HexPlane-DepthSupervision-depth-pro/vanilla": "Depth Supervision (Depth Pro)",
    "HexPlane-DepthSupervision-mega-sam/vanilla": "Depth Supervision (MegaSaM)",
    "HexPlane-DepthSupervision+GaussianInit-videoda/vanilla": "GaussianInit (VideoDA)",
    "HexPlane-DepthSupervision+GaussianInit-depth-pro/vanilla": "GaussianInit (Depth Pro)",
    "HexPlane-DepthSupervision+GaussianInit-mega-sam/vanilla": "GaussianInit (MegaSaM)",
    "HexPlane/vanilla": "4D-GS",
}

splits = ["1"] #, "2", "3"]

# Hochkant layout
normal_positions = {
    (0, 0): "GT",
    (0, 1): "MLP/vanilla",
    (1, 1): "MLP-DepthSupervision-videoda/vanilla",
    (2, 1): "MLP-DepthSupervision-depth-pro/vanilla",
    (3, 1): "MLP-DepthSupervision+GaussianInit-videoda/vanilla",
    (4, 1): "MLP-DepthSupervision+GaussianInit-depth-pro/vanilla",
    (0, 2): "Curve/vanilla",
    (1, 2): "Curve-DepthSupervision-videoda/vanilla",
    (2, 2): "Curve-DepthSupervision-depth-pro/vanilla",
    (3, 2): "Curve-DepthSupervision+GaussianInit-videoda/vanilla",
    (4, 2): "Curve-DepthSupervision+GaussianInit-depth-pro/vanilla",
    (0, 3): "HexPlane/vanilla",
    (1, 3): "HexPlane-DepthSupervision-videoda/vanilla",
    (2, 3): "HexPlane-DepthSupervision-depth-pro/vanilla",
    (3, 3): "HexPlane-DepthSupervision+GaussianInit-videoda/vanilla",
    (4, 3): "HexPlane-DepthSupervision+GaussianInit-depth-pro/vanilla",
}

positions = {
    (0, 0): "GT",
    (1, 0): "MLP/vanilla",
    (2, 0): "MLP-DepthSupervision-videoda/vanilla",
    (3, 0): "MLP-DepthSupervision-depth-pro/vanilla",
    (4, 0): "MLP-DepthSupervision-mega-sam/vanilla",
    (5, 0): "MLP-DepthSupervision+GaussianInit-videoda/vanilla",
    (6, 0): "MLP-DepthSupervision+GaussianInit-depth-pro/vanilla",
    (7, 0): "MLP-DepthSupervision+GaussianInit-mega-sam/vanilla",
    (0, 1): "GT",
    (1, 1): "Curve/vanilla",
    (2, 1): "Curve-DepthSupervision-videoda/vanilla",
    (3, 1): "Curve-DepthSupervision-depth-pro/vanilla",
    (4, 1): "Curve-DepthSupervision-mega-sam/vanilla",
    (5, 1): "Curve-DepthSupervision+GaussianInit-videoda/vanilla",
    (6, 1): "Curve-DepthSupervision+GaussianInit-depth-pro/vanilla",
    (7, 1): "Curve-DepthSupervision+GaussianInit-mega-sam/vanilla",
    (0, 2): "GT",
    (1, 2): "HexPlane/vanilla",
    (2, 2): "HexPlane-DepthSupervision-videoda/vanilla",
    (3, 2): "HexPlane-DepthSupervision-depth-pro/vanilla",
    (4, 2): "HexPlane-DepthSupervision-mega-sam/vanilla",
    (5, 2): "HexPlane-DepthSupervision+GaussianInit-videoda/vanilla",
    (6, 2): "HexPlane-DepthSupervision+GaussianInit-depth-pro/vanilla",
    (7, 2): "HexPlane-DepthSupervision+GaussianInit-mega-sam/vanilla",
}
# Quer layout
# positions = {
#     (0, 0): "GT",
#     (1, 0): "Curve/vanilla",
#     (1, 1): "Curve-videoda/vanilla",
#     (2, 0): "MLP/vanilla",
#     (2, 1): "MLP-videoda/vanilla",
#     (3, 0): "HexPlane/vanilla",
#     (3, 1): "HexPlane-videoda/vanilla",
# }


fps = {
    # iphone
    "apple": 30,
    "backpack": 30,
    "block": 30,
    "creeper": 30,
    "handwavy": 30,
    "haru-sit": 30,
    "mochi-high-five": 30,
    "paper-windmill": 30,
    "pillow": 30,
    "spin": 30,
    "sriracha-tree": 30,
    "teddy": 30,
    # nerfies
    "curls": 5,
    "broom": 5,
    "tail": 15,
    "toby-sit": 15,
    # unknown
    "espresso": 15,
    "torchocolate": 15,
    "vrig-peel-banana": 15,
    "as": 30,
    "plate": 30,
    "sieve": 30,
    # dnerf
    "bouncingballs": 1,
    "standup": 1,
    "trex": 1,
}


root_dir = "../../output/depth_experiment"
tineuvox_root_dir = "../../TiNeuVox/logs"
exp_prefix = "website_videos_masked_wo_depth_compact"
os.makedirs(exp_prefix, exist_ok=True)
max_length = 100

import os
import numpy as np
import cv2


for dataset in tqdm(dataset_mapper):
    os.makedirs(os.path.join(exp_prefix, dataset_mapper[dataset]), exist_ok=True)
    for scene in tqdm(scenes[dataset]):
        video_path = os.path.join(exp_prefix, dataset_mapper[dataset], scene + ".mp4")
        if os.path.exists(video_path):
            print(f"Existing video_path {video_path}!")
# continue
        print(f"Preparing for video_path {video_path}")
        # get all video readers
        # video_readers = {}
        gt_video_path = os.path.join(root_dir, dataset, scene, "Curve-DepthSupervision-videoda/vanilla1", "test_mask.mp4")
        video_readers = {}
        video_readers["GT"] = imageio.get_reader(gt_video_path, "mp4", fps=10)
        for method in method_mapper:
            # select the video that has the highest psnr
            max_psnr = -1
            for split in splits:
                if method.startswith("TiNeuVox"):
                    log_path = os.path.join(tineuvox_root_dir, dataset, scene, "vanilla" + split)
                else:
                    log_path = os.path.join(root_dir, dataset, scene, method + split)
                with open(os.path.join(log_path, "test.txt"), "r") as f:
                    line = f.readline()
                    while line:
                        if line.startswith("Average PSNR:"):
                            test_psnr = float(line.strip().split(" ")[-1])
                        line = f.readline()
                # check if test_psnr == nan
                if (test_psnr > max_psnr) or ((np.isnan(test_psnr) and max_psnr == -1)):
                    video_reader_path = os.path.join(log_path, "test_mask.mp4")
            print(f"Selected video_reader_path {video_reader_path} with PSNR {test_psnr}")
            video_readers[method] = imageio.get_reader(video_reader_path, "mp4", fps=10)
#             if "GT" not in video_readers:
#                 video_readers["GT"] = imageio.get_reader(video_reader_path, "mp4", fps=10)
        first_video = next(iter(video_readers.values()))
        frame_width, frame_height = (
            first_video.get_next_data().shape[1],
            first_video.get_next_data().shape[0],
        )
        first_video.set_image_index(0)  # reset to the first frame
        frame_width = frame_width // 4
        frame_height = frame_height // 2
        # Calculate the dimensions of the stitched video
        grid_width, grid_height = (
            max(positions, key=lambda x: x[0])[0] + 1,
            max(positions, key=lambda x: x[1])[1] + 1,
        )
        stitched_width, stitched_height = (
            frame_width * grid_width,
            frame_height * grid_height,
        )

        # Create a writer for the stitched video
        writer = imageio.get_writer(video_path, fps=fps[scene])
        cur_index = 0
        while True:
            stitched_frame = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
            
            # Read GT frame once and store it for reuse
            gt_frame_cache = None
            
            for position, video_key in positions.items():
                col, row = position
                video = video_readers[video_key]
                try:
                    # If this is GT and we already read it, use cached frame
                    if video_key == "GT":
                        if gt_frame_cache is None:
                            # First GT position - read the frame
                            frame = video.get_next_data()
                            if teleport[scene]:
                                frame = video.get_next_data()
                            gt_frame_cache = frame.copy()
                        else:
                            # Subsequent GT positions - use cached frame
                            frame = gt_frame_cache.copy()
                    else:
                        # Non-GT methods - read normally
                        frame = video.get_next_data()
                        if teleport[scene]:
                            frame = video.get_next_data()
                    
                    # Extract only RGB (no depth)
                    if video_key == "GT":
                        rgb = frame[:frame_height, :frame_width]
                    else:
                        rgb = frame[:frame_height, frame_width : 2 * frame_width]
                    frame = rgb  # Use only RGB, no depth concatenation
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(
                        frame,
                        (frame_width, frame_height),
                        interpolation=cv2.INTER_AREA,
                    )

                    # Add method name to the top left corner of the frame
                    method_name = method_mapper.get(video_key, "")
                    if video_key == "GT":
                        method_name = "GT"
                    text_size, _ = cv2.getTextSize(method_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_width, text_height = text_size
                    bg_left = 5
                    bg_top = 5
                    bg_right = bg_left + text_width + 10
                    bg_bottom = bg_top + text_height + 10
                    frame = cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
                    frame = cv2.putText(
                        frame,
                        method_name,
                        (bg_left + 5, bg_bottom - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # Convert frame back to RGB color space
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stitched_frame[
                        row * frame_height : (row + 1) * frame_height,
                        col * frame_width : (col + 1) * frame_width,
                    ] = frame
                except (IndexError, StopIteration):
                    # If the video has ended, reset its reader
                    stitched_frame = None
                    break
                    # video.set_image_index(0)
                    # frame = video.get_next_data()

            if stitched_frame is not None:
                writer.append_data(stitched_frame)

            # Break the loop if any video has ended
            if not all(video.get_length() > cur_index for video in video_readers.values()):
                break

            # Break the loop if max video length is reached
            if not all(max_length > cur_index for video in video_readers.values()):
                break
            cur_index += 1

        writer.close()

        # destroy all video readers
        for method in video_readers:
            video_readers[method].close()
