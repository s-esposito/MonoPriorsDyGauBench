import os
import shutil
import multiprocessing
from functools import partial

def compress_scene(scene_path, dest_dir):
    scene_name = os.path.basename(scene_path)
    zip_path = os.path.join(dest_dir, scene_name)
    #print(zip_path, scene_path)
    shutil.make_archive(zip_path, 'zip', scene_path)
    
    return f"Compressed {scene_name} from {scene_path} to {zip_path}"

def process_dataset(dataset_path, dest_path):
    dataset_name = os.path.basename(dataset_path)
    dest_dir = os.path.join(dest_path, dataset_name)
    os.makedirs(dest_dir, exist_ok=True)

    if dataset_name == 'dnerf':
        data_path = os.path.join(dataset_path, 'data')
        scenes = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        dest_dir = os.path.join(dest_dir, 'data')  # Adjust destination for dnerf
    else:
        scenes = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    os.makedirs(dest_dir, exist_ok=True)

    with multiprocessing.Pool() as pool:
        results = pool.map(partial(compress_scene, dest_dir=dest_dir), scenes)
    
    for result in results:
        print(result)

def compress_folder(source_folder, destination_folder, datasets):
    os.makedirs(destination_folder, exist_ok=True)

    datasets = [os.path.join(source_folder, d) for d in datasets if os.path.isdir(os.path.join(source_folder, d))]

    for dataset in datasets:
        process_dataset(dataset, destination_folder) 

if __name__ == "__main__":
    source_path = 'data'
    destination_path = 'data_compressed'
    
    #datasets = ['dnerf']
    datasets = ['nerfds', 'nerfies']

    compress_folder(source_path, destination_path, datasets)

    print("Compression completed successfully!")