import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tqdm import tqdm
from scipy.ndimage import sobel
import pickle
import os
import json

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    image = Image.open(image_path).convert('L')
    width, height = image.size
    max_edge = max(width, height)
    scale = 480 / max_edge
    new_size = (int(width * scale), int(height * scale))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(image)

def compute_magnitude_spectrum(image_array):
    """Compute the magnitude spectrum of an image."""
    fft_image = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.abs(fft_shifted)
    return magnitude_spectrum


def aggregate_magnitude_spectrums(image_paths, batch_size=10, sample_rate=1):
    """Aggregate magnitude spectrums from a list of images."""
    aggregated_magnitudes = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        for image_path in batch_paths:
            image_array = load_image(image_path)
            magnitude_spectrum = compute_magnitude_spectrum(image_array)
            magnitude_spectrum = magnitude_spectrum.flatten()
            #magnitude_sum = np.sum(magnitude_spectrum)
            aggregated_magnitudes.append(magnitude_spectrum)
            #np.random.shuffle(magnitude_spectrum)
            #subsample = magnitude_spectrum[::sample_rate]
            #aggregated_magnitudes.extend(subsample)

    return np.array(aggregated_magnitudes)

def plot_histogram(aggregated_magnitudes, outpath, bins=50):
    """Plot histogram of the aggregated magnitude spectrums."""
    min_value = np.percentile(aggregated_magnitudes, 1)
    max_value = np.percentile(aggregated_magnitudes, 99)
    mean_value = np.mean(aggregated_magnitudes)

    plt.figure(figsize=(10, 6))
    #plt.hist(aggregated_magnitudes, bins=bins, range=(min_value, max_value), log=True, edgecolor='black')
    plt.axvline(mean_value, color='r', linestyle='dashed', linewidth=1)
    plt.text(mean_value * 1.1, plt.ylim()[1] * 0.9, 'Mean: {:.2f}'.format(mean_value), color='r')
   
    plt.title('Aggregated Magnitude Spectrum Distribution')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True)
    plt.savefig(outpath)
    plt.close()

# Example usage
#image_folder = 'data/hypernerf/americano/rgb/1x/*.png'  # Replace with your image folder path
root_dir = ".."
save_dir = "freq"
os.makedirs(save_dir, exist_ok=True)

image_folders = {
    "dnerf/bouncingballs": "data/dnerf/data/bouncingballs/train/*.png",
    "dnerf/hellwarrior": "data/dnerf/data/hellwarrior/train/*.png",
    "dnerf/hook": "data/dnerf/data/hook/train/*.png",
    "dnerf/jumpingjacks": "data/dnerf/data/jumpingjacks/train/*.png",
    "dnerf/lego": "data/dnerf/data/lego/train/*.png",
    "dnerf/mutant": "data/dnerf/data/mutant/train/*.png",
    "dnerf/standup": "data/dnerf/data/standup/train/*.png",
    "dnerf/trex": "data/dnerf/data/trex/train/*.png",
    "nerfds/as": 'data/nerfds/as/rgb/1x/*.png',
    "nerfds/basin": 'data/nerfds/basin/rgb/1x/*.png',
    "nerfds/bell": 'data/nerfds/bell/rgb/1x/*.png',
    "nerfds/cup": 'data/nerfds/cup/rgb/1x/*.png',
    "nerfds/plate": 'data/nerfds/plate/rgb/1x/*.png',
    "nerfds/press": 'data/nerfds/press/rgb/1x/*.png',
    "nerfds/sieve": 'data/nerfds/sieve/rgb/1x/*.png',
    "nerfies/broom": 'data/nerfies/broom/rgb/2x/*.png',
    "nerfies/curls": 'data/nerfies/curls/rgb/4x/*.png',
    "nerfies/tail": 'data/nerfies/tail/rgb/2x/*.png',
    "nerfies/toby-sit": 'data/nerfies/toby-sit/rgb/2x/*.png',
    "hypernerf/aleks-teapot": "data/hypernerf/aleks-teapot/rgb/2x/*.png",
    "hypernerf/americano": "data/hypernerf/americano/rgb/2x/*.png",
    "hypernerf/broom2": "data/hypernerf/broom2/rgb/2x/*.png",
    "hypernerf/chickchicken": "data/hypernerf/chickchicken/rgb/2x/*.png",
    "hypernerf/cross-hands1": "data/hypernerf/cross-hands1/rgb/2x/*.png",
    "hypernerf/cut-lemon1": "data/hypernerf/cut-lemon1/rgb/2x/*.png",
    "hypernerf/espresso": "data/hypernerf/espresso/rgb/2x/*.png",
    "hypernerf/hand1-dense-v2": "data/hypernerf/hand1-dense-v2/rgb/2x/*.png",
    "hypernerf/keyboard": "data/hypernerf/keyboard/rgb/2x/*.png",
    "hypernerf/oven-mitts": "data/hypernerf/oven-mitts/rgb/2x/*.png",
    "hypernerf/slice-banana": "data/hypernerf/slice-banana/rgb/2x/*.png",
    "hypernerf/split-cookie": "data/hypernerf/split-cookie/rgb/2x/*.png",
    "hypernerf/tamping": "data/hypernerf/tamping/rgb/4x/*.png",
    "hypernerf/torchocolate": "data/hypernerf/torchocolate/rgb/2x/*.png",
    "hypernerf/vrig-3dprinter": "data/hypernerf/vrig-3dprinter/rgb/2x/*.png",
    "hypernerf/vrig-chicken": "data/hypernerf/vrig-chicken/rgb/2x/*.png",
    "hypernerf/vrig-peel-banana": "data/hypernerf/vrig-peel-banana/rgb/2x/*.png",
    "iphone/apple": "data/iphone/apple/rgb/2x/*.png",
    "iphone/backpack": "data/iphone/backpack/rgb/2x/*.png",
    "iphone/block": "data/iphone/block/rgb/2x/*.png",
    "iphone/creeper": "data/iphone/creeper/rgb/2x/*.png",
    "iphone/handwavy": "data/iphone/handwavy/rgb/2x/*.png",
    "iphone/haru-sit": "data/iphone/haru-sit/rgb/2x/*.png",
    "iphone/mochi-high-five": "data/iphone/mochi-high-five/rgb/2x/*.png",
    "iphone/paper-windmill": "data/iphone/paper-windmill/rgb/2x/*.png",
    "iphone/pillow": "data/iphone/pillow/rgb/2x/*.png",
    "iphone/space-out": "data/iphone/space-out/rgb/2x/*.png",
    "iphone/spin": "data/iphone/spin/rgb/2x/*.png",
    "iphone/sriracha-tree": "data/iphone/sriracha-tree/rgb/2x/*.png",
    "iphone/teddy": "data/iphone/teddy/rgb/2x/*.png",
    "iphone/wheel": "data/iphone/wheel/rgb/2x/*.png",
    
}


batch_size = 10

results = {}
for scene_name, image_folder in image_folders.items():
    dataset, scene = scene_name.split("/")
    if dataset not in results:
        results[dataset] = {}
    
    image_paths = glob.glob(os.path.join(root_dir, image_folder))
   

    #if os.path.exists(f"{save_dir}/{dataset}_{scene}.pkl"):
        #with open(f'{save_dir}/{dataset}_{scene}.pkl', 'rb') as file:
        #    aggregated_magnitudes = pickle.load(file)
    #else:
    aggregated_magnitudes = aggregate_magnitude_spectrums(image_paths, batch_size)    
        #with open(f'{save_dir}/{dataset}_{scene}.pkl', 'wb') as file:
        #    pickle.dump(aggregated_magnitudes, file)
    outpath = f"{save_dir}/{dataset}_{scene}.png"
    plot_histogram(aggregated_magnitudes, outpath)

    mean_value = np.mean(aggregated_magnitudes)
    results[dataset][scene] = mean_value

    with open("freq/freq.json", "w") as outfile:
        json.dump(results, outfile, indent=4)

#for dataset in results:
#    means = results[dataset]
#    results[dataset] = sum(means)/float(len(means))

#for dataset in  results:
#    print(dataset, results[dataset])
