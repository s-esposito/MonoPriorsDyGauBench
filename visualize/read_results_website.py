import imageio
import numpy as np
import cv2
import os
from tqdm import tqdm
import copy

dataset_mapper = {
    "dnerf": "DNeRF",
    "hypernerf": "HyperNeRF",
    "iphone": "iPhone",
    "nerfds": "NeRF-DS",
    "nerfies": "Nerfies",
}

scenes = {
    "dnerf": ["bouncingballs", "standup", "trex"],
    "hypernerf": ["espresso", "torchocolate", "vrig-peel-banana"],
    "iphone": ["apple", "mochi-high-five", "paper-windmill"],
    "nerfds": ["as", "plate", "sieve"],
    "nerfies": ["curls", "tail", "toby-sit"],
}

teleport = {
    "bouncingballs": False,
    "standup": False,
    "trex": False,
    "espresso": False,
    "torchocolate": False,
    "vrig-peel-banana": True,
    "apple": False,
    "mochi-high-five": False,
    "paper-windmill": False,
    "curls": True,
    "tail": True,
    "toby-sit": True,
}


method_mapper = {
    "TiNeuVox/vanilla": "TiNeuVox",
    "MLP/nodeform": "3DGS",
    "MLP/vanilla": "DeformableGS",
    "HexPlane/vanilla": "4DGS",
    "FourDim/vanilla": "RTGS",
    "TRBF/nodecoder": "STG_nodecoder",
    "TRBF/vanilla": "STG",
}

splits = ["1", "2", "3"]

positions = {
    (0, 0): "GT",
    # (0, 1): "TiNeuVox/vanilla",
    (1, 0): "MLP/nodeform",
    (1, 1): "MLP/vanilla",
    (2, 0): "HexPlane/vanilla",
    (2, 1): "FourDim/vanilla",
    (3, 0): "TRBF/nodecoder",
    (3, 1): "TRBF/vanilla",
}

fps = {
    "apple": 30,
    "mochi-high-five": 60,
    "paper-windmill": 30,
    "curls": 5,
    "tail": 15,
    "toby-sit": 15,
    "espresso": 15,
    "torchocolate": 15,
    "vrig-peel-banana": 15,
    "as": 30,
    "plate": 30,
    "sieve": 30,
    "bouncingballs": 1,
    "standup": 1,
    "trex": 1,
}


root_dir = "../output"
tineuvox_root_dir = "../../TiNeuVox/logs"
exp_prefix = "website_videos"
os.makedirs(exp_prefix, exist_ok=True)
max_length = 100


for dataset in tqdm(dataset_mapper):
    os.makedirs(os.path.join(exp_prefix, dataset_mapper[dataset]), exist_ok=True)
    for scene in tqdm(scenes[dataset]):
        video_path = os.path.join(exp_prefix, dataset_mapper[dataset], scene + ".mp4")
        if os.path.exists(video_path):
            print(f"Existing video_path {video_path}!")
            continue
        print(f"Preparing for video_path {video_path}")
        # get all video readers
        video_readers = {}
        for method in method_mapper:
            # select the video that has the highest psnr
            max_psnr = -1
            for split in splits:
                if method.startswith("TiNeuVox"):
                    log_path = os.path.join(
                        tineuvox_root_dir, dataset, scene, "vanilla" + split
                    )
                else:
                    log_path = os.path.join(root_dir, dataset, scene, method + split)
                with open(os.path.join(log_path, "test.txt"), "r") as f:
                    line = f.readline()
                    while line:
                        if line.startswith("Average PSNR:"):
                            test_psnr = float(line.strip().split(" ")[-1])
                        line = f.readline()
                if test_psnr > max_psnr:
                    video_reader_path = os.path.join(log_path, "test.mp4")
            video_readers[method] = imageio.get_reader(video_reader_path, "mp4", fps=10)
            if "GT" not in video_readers:
                video_readers["GT"] = imageio.get_reader(
                    video_reader_path, "mp4", fps=10
                )
        first_video = next(iter(video_readers.values()))
        frame_width, frame_height = (
            first_video.get_next_data().shape[1],
            first_video.get_next_data().shape[0],
        )
        frame_width = frame_width // 3
        frame_height = frame_height // 2
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
        writer = imageio.get_writer(video_path, fps=fps[scene])
        cur_index = 0
        while True:
            stitched_frame = np.zeros(
                (stitched_height, stitched_width, 3), dtype=np.uint8
            )

            for position, video_key in positions.items():
                col, row = position
                video = video_readers[video_key]
                try:
                    frame = video.get_next_data()
                    if teleport[scene]:
                        frame = video.get_next_data()
                    if video_key == "GT":
                        rgb = frame[:frame_height, :frame_width]
                        depth = np.zeros_like(rgb)
                    else:
                        rgb = frame[:frame_height, frame_width : 2 * frame_width]
                        depth = frame[:frame_height, 2 * frame_width :]
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
                    text_size, _ = cv2.getTextSize(
                        method_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    text_width, text_height = text_size
                    bg_left = 5
                    bg_top = 5
                    bg_right = bg_left + text_width + 10
                    bg_bottom = bg_top + text_height + 10
                    frame = cv2.rectangle(
                        frame, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1
                    )
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

            # Break the loop if any video has ended
            if not all(
                video.get_length() > cur_index for video in video_readers.values()
            ):
                break

            # Break the loop if max video length is reached
            if not all(max_length > cur_index for video in video_readers.values()):
                break
            cur_index += 1

        writer.close()

        # destroy all video readers
        for method in video_readers:
            video_readers[method].close()
