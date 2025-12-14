import imageio
import numpy as np
import cv2
import os
from tqdm import tqdm
import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt

dataset_mapper = {
    "nerfies": "Nerfies",
}

scenes = {
    # "nerfies": ["broom", "curls", "tail", "toby-sit"]
    "nerfies": ["curls"]
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
    "Curve-DepthSupervision-videoda/vanilla": "GT_RGB",
    "Curve-DepthSupervision-videoda/vanilla": "GT_DEPTH_VIDEODA",
    "Curve-DepthSupervision-depth-pro/vanilla": "GT_DEPTH_DEPTHPRO",
    
    "Curve-DepthSupervision-videoda/vanilla": "EffGS + Depth Supervision (VideoDA)", # <---- it is important that a depth method is the first one here !!!
    "Curve-DepthSupervision-depth-pro/vanilla": "EffGS + Depth Supervision (Depth Pro)",
    "Curve/vanilla": "EffGS",
    
    "MLP-DepthSupervision-videoda/vanilla": "DeformableGS + Depth Supervision (VideoDA)",
    "MLP-DepthSupervision-depth-pro/vanilla": "DeformableGS + Depth Supervision (Depth Pro)",
    "MLP/vanilla": "DeformableGS",
    
    "HexPlane-DepthSupervision-videoda/vanilla": "4D-GS + Depth Supervision (VideoDA)",
    "HexPlane-DepthSupervision-depth-pro/vanilla": "4D-GS + Depth Supervision (Depth Pro)",
    "HexPlane/vanilla": "4D-GS",
}

splits = ["1"] #, "2", "3"]


# Hochkant layout
positions = {
    (0, 0): "GT_RGB",
    (1, 0): "GT_DEPTH_VIDEODA",
    (2, 0): "GT_DEPTH_DEPTHPRO",
    (0, 1): "MLP/vanilla",
    (1, 1): "MLP-DepthSupervision-videoda/vanilla",
    (2, 1): "MLP-DepthSupervision-depth-pro/vanilla",
    (0, 2): "Curve/vanilla",
    (1, 2): "Curve-DepthSupervision-videoda/vanilla",
    (2, 2): "Curve-DepthSupervision-depth-pro/vanilla",
    (0, 3): "HexPlane/vanilla",
    (1, 3): "HexPlane-DepthSupervision-videoda/vanilla",
    (2, 3): "HexPlane-DepthSupervision-depth-pro/vanilla",
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
    "broom": 30,
    "tail": 30,
    "toby-sit": 30,
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
exp_prefix = "website_videos"
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
        video_readers = {}
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
                    video_reader_path = os.path.join(log_path, "test.mp4")
            print(f"Selected video_reader_path {video_reader_path} with PSNR {test_psnr}")
            video_readers[method] = imageio.get_reader(video_reader_path, "mp4", fps=10)
            if "GT" not in video_readers:
                video_readers["GT"] = imageio.get_reader(video_reader_path, "mp4", fps=10)
            if "GT_RGB" not in video_readers:
                video_readers["GT_RGB"] = imageio.get_reader(video_reader_path, "mp4", fps=10)
            if "GT_DEPTH_VIDEODA" not in video_readers:
                video_readers["GT_DEPTH_VIDEODA"] = imageio.get_reader(video_reader_path, "mp4", fps=10)
            if "GT_DEPTH_DEPTHPRO" not in video_readers:
                log_path = os.path.join(root_dir, dataset, scene, "Curve-DepthSupervision-depth-pro/vanilla" + split)
                video_reader_path = os.path.join(log_path, "test.mp4")
                video_readers["GT_DEPTH_DEPTHPRO"] = imageio.get_reader(video_reader_path, "mp4", fps=10)
        first_video = next(iter(video_readers.values()))
        frame_width, frame_height = (
            first_video.get_next_data().shape[1],
            first_video.get_next_data().shape[0],
        )
        first_video.set_image_index(0)  # reset to the first frame
        frame_width = frame_width // 4
        frame_height = frame_height // 2
        # # Add this to reduce further:
        # frame_width = frame_width // 2  # or frame_width = int(frame_width * 0.75)
        # frame_height = frame_height // 2
        # Calculate the dimensions of the stitched video
        grid_width, grid_height = (
            max(positions, key=lambda x: x[0])[0] + 1,
            max(positions, key=lambda x: x[1])[1] + 1,
        )
        stitched_width, stitched_height = (
            frame_width * grid_width * 2,
            frame_height * grid_height,
        )

        # Create a writer for the stitched video
        # writer = imageio.get_writer(video_path, fps=fps[scene])# , quality=1)
        # writer = imageio.get_writer(
        #     video_path, 
        #     fps=fps[scene],
        #     codec='libx264',
        #     # quality=10,
        #     pixelformat='yuv420p',
        #     macro_block_size=16,
        #     ffmpeg_params=[
        #         '-crf', '25',  # Lower = better quality (15-23 is good, default is 23)
        #         '-preset', 'slow',  # Better compression efficiency
        #         '-profile:v', 'high',  # H.264 high profile
        #         '-pix_fmt', 'yuv420p'  # Compatibility
        #     ]
        # )
        
#         writer = imageio.get_writer(
#             video_path, 
#             fps=fps[scene],
#             codec='libx264',
#             pixelformat='yuv420p',
#             macro_block_size=16,
#             bitrate='80000k',  # Target bitrate instead of CRF
#             ffmpeg_params=[
#                 '-preset', 'slow',
#                 '-profile:v', 'high',
#                 '-bufsize', '4000k'
#             ]
#         )
        # writer = imageio.get_writer(
        #     # video_path.replace('.mp4', '.webm'),  # VP9 typically uses .webm
        #     video_path.replace('.webm', '.mp4'),
        #     fps=fps[scene],
        #     codec='libvpx-vp9',
        #     pixelformat='yuv420p',
        #     quality=8,
        # )
        # writer = imageio.get_writer(
        #     video_path, 
        #     fps=fps[scene],
        #     codec='libx264',
        #     pixelformat='yuv420p',
        #     ffmpeg_params=[ '-preset', 'slow', '-crf', '17' ]
        # )
        # writer = imageio.get_writer(
        #     video_path,
        #     fps=fps[scene],
        #     codec='libx265',
        #     pixelformat='yuv420p',
        #     ffmpeg_params=[
        #         '-preset', 'medium',              # medium keeps size reasonable
        #         '-x265-params', 'crf=23:aq-mode=3',
        #         '-movflags', '+faststart'
        #     ]
        # )
        # writer = imageio.get_writer(
        #     video_path,
        #     fps=fps[scene],
        #     codec="libx264",
        #     pixelformat="yuv420p",
        #     ffmpeg_params=[
        #         "-preset", "medium",
        #         "-crf", "23",               # This is the KEY change
        #         "-profile:v", "high",
        #         "-tune", "animation",       # MUCH smaller than grain
        #         "-movflags", "+faststart"
        #     ]
        # )
        writer = imageio.get_writer(
            video_path,  # must be .mp4
            fps=fps[scene],
            codec='libx264',
            pixelformat='yuv420p',
            ffmpeg_params=[
                '-preset', 'veryslow',     # or 'slow' if too slow to encode
                '-crf', '28',              # 28 = tiny but gorgeous for renders
                '-tune', 'animation',      # critical for clean synthetic content
                '-profile:v', 'high',
                '-level', '4.2',           # ensures broad compatibility
                '-movflags', '+faststart', # progressive download for web
                '-bf', '0',                # fixes thumbnail/seeking issues on macOS
            ]
        )
        cur_index = 0
        while True:
            stitched_frame = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)
            black_panel = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            for position, video_key in positions.items():
                col, row = position
                video = video_readers[video_key]
                try:
                    frame = video.get_next_data()
                    if teleport[scene]:
                        frame = video.get_next_data()
                    # print(video_key)
                    # print(position)
                    # print("-" in video_key)
                    # presave depth
                    if video_key == "GT" or video_key == "GT_RGB":
                        rgb = frame[:frame_height, :frame_width]
                        depth = black_panel #frame[:frame_height, 2 * frame_width : 3 * frame_width]
                    elif video_key == "GT_DEPTH_VIDEODA" or video_key == "GT_DEPTH_DEPTHPRO" or video_key == "GT_DEPTH_MEGASAM":
                        rgb = black_panel #frame[:frame_height, :frame_width]
                        depth = frame[:frame_height, 2 * frame_width : 3 * frame_width]
                    else:
                        rgb = frame[:frame_height, frame_width : 2 * frame_width]
                        depth = frame[:frame_height, 3 * frame_width :]    
                    frame = np.concatenate([rgb, depth], axis=1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.resize(
                        frame,
                        (frame_width * 2, frame_height),
                        interpolation=cv2.INTER_AREA,
                    )

                    # Add method name to the top left corner of the frame
                    method_name = method_mapper.get(video_key, "")
                    if video_key == "GT":
                        method_name = "GT"
                    if video_key == "GT_RGB":
                        method_name = "GT"
                    if video_key == "GT_DEPTH_VIDEODA":
                        method_name = "GT Depth (VideoDA)"
                    if video_key == "GT_DEPTH_DEPTHPRO":
                        method_name = "GT Depth (Depth Pro)"
                    if video_key == "GT_DEPTH_MEGASAM":
                        method_name = "GT Depth (MegaSaM)"
                    text_size, _ = cv2.getTextSize(method_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_width, text_height = text_size
                    if video_key.startswith("GT_DEPTH"):
                        bg_left = frame_width + 5
                    else:
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
                        col * frame_width * 2 : (col + 1) * frame_width * 2,
                    ] = frame
                except (IndexError, StopIteration):
                    # If the video has ended, reset its reader
                    stitched_frame = None
                    break
                    # video.set_image_index(0)
                    # frame = video.get_next_data()

            if stitched_frame is not None:
                writer.append_data(stitched_frame)
            
#             # Inside the main loop, after creating stitched_frame
#             debug_frame = cv2.resize(stitched_frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
#             test_path = f"debug_frame_{scene}_{cur_index}.png"
#             imageio.imwrite(test_path, debug_frame)
#             print(f"Saved debug frame: {test_path}")

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
