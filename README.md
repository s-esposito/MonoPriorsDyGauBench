# Monocular Dynamic Gaussian Splatting: Fast, Brittle, and Scene Complexity Rules

This repository is the official PyTorch(Lightning) implementation of the paper:

[**Monocular Dynamic Gaussian Splatting: Fast, Brittle, and Scene Complexity Rules**](https://lynl7130.github.io/MonoDyGauBench.github.io/)  
[Yiqing Liang](https://lynl7130.github.io), [Mikhail Okunev](https://mmehas.github.io/), [Mikaela Angelina Uy](https://mikacuy.github.io/)†‡, [Runfeng Li](https://www.linkedin.com/in/runfeng-l-a41b6a204/), [Leonidas Guibas](https://profiles.stanford.edu/leonidas-guibas)‡, [James Tompkin](https://jamestompkin.com/), [Adam W Harley](https://adamharley.com/)‡  

<img width="12%"  text-align="center" margin="auto" src=images/brownlogo.svg> &nbsp;&nbsp;
†<img width="20%"  text-align="center" margin="auto" src=images/nvidia_logo.png>  &nbsp;&nbsp;
‡<img width="8%"  text-align="center" margin="auto" src=images/stanfordlogo.png>

&nbsp;&nbsp;[Paper](https://lynl7130.github.io/data/DyGauBench_tmp.pdf) &nbsp;| &nbsp;[Data](https://1drv.ms/f/c/4dd35d8ee847a247/EpmindtZTxxBiSjYVuaaiuUBr7w3nOzEl6GjrWjmVPuBFw?e=cW5gg1)

We aim to <strong>benchmark</strong> Monocular View Dynamic Gaussian Splatting from motion perspective.

### Methods Included
| Method                                              | Abbrev Name in this Repo |
| ------------------------------------------------- | ---- |
<input type="checkbox" disabled checked />  [Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting](https://arxiv.org/abs/2310.10642) (ICLR 2024) | FourDim |
<input type="checkbox" disabled checked /> [A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](https://arxiv.org/pdf/2311.12897) (ECCV 2024) | Curve |
<input type="checkbox" disabled checked /> [4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://arxiv.org/abs/2310.08528) (CVPR 2024)| HexPlane |
<input type="checkbox" disabled checked /> [Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://arxiv.org/abs/2309.13101) (CVPR 2024) | MLP |
<input type="checkbox" disabled checked /> [Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis](https://arxiv.org/abs/2312.16812) (CVPR 2024) | TRBF |


### Existing Datasets Included 

| Dataset                                              | Abbrev Name in this Repo |
| ------------------------------------------------- | ---- |
<input type="checkbox" disabled checked /> [D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961) (CVPR 2021) | dnerf |
<input type="checkbox" disabled checked /> [Nerfies: Deformable Neural Radiance Fields](https://arxiv.org/abs/2011.12948) (ICCV 2021) | nerfies |
<input type="checkbox" disabled checked /> [A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields](https://arxiv.org/abs/2106.13228) (SIGGRAPH Asia 2021) | hypernerf |
<input type="checkbox" disabled checked /> [Monocular Dynamic View Synthesis: A Reality Check](https://arxiv.org/abs/2210.13445) (NeurIPS 2022) | iphone |
<input type="checkbox" disabled checked /> [NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects](https://arxiv.org/abs/2303.14435) (CVPR 2023) | nerfds |

### Our Created Datasets
| Dataset                                              | Abbrev Name in this Repo |
| ------------------------------------------------- | ---- |
Camera-Pose-Rectified HyperNeRF | fixed |
Instructive Dataset | dnerf/custom |

All data could be prepared by downloading [this folder](https://1drv.ms/f/c/4dd35d8ee847a247/EpmindtZTxxBiSjYVuaaiuUBr7w3nOzEl6GjrWjmVPuBFw?e=cW5gg1) and extracted as follow diagram:
```
this_repo
│   README.md  
└───data
│    │ 
│    └───dnerf
│    │    │ 
│    │    └───data
│    │         │ 
│    │         └───bouncingballs
│    │         └───...
│    └───fixed
│    │    │ 
│    │    └───chickchicken
│    │    └───...
│    └───hypernerf 
│    │    │ 
│    │    └───aleks-teapot
│    │    └───...
│    └───iphone
│    │    │ 
│    │    └───apple
│    │    └───...
│    └───nerfds
│    │    │ 
│    │    └───as
│    │    └───...
│    └───nerfies
│    │     │ 
│    │     └───broom
│    │     └───...
│    │
|    └───custom
|          |
│          └───dynamic_cube_dynamic_camera_textured_motion_range_0.0
│          └───...
└...       
...
```

## Installation
This code has been developed with Anaconda (Python 3.9), CUDA 12.4.1, one NVIDIA GeForce RTX 4090 GPU.  

```Shell
conda create -n monodybench python=3.9
conda activate monodybench
# conda installs
conda install nvidia/label/cuda-12.4.1::cuda-toolkit
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install -c conda-forge plyfile==0.8.1
conda install conda-forge::glm

# pip installs
pip install -r requirements.txt

# install from local folders 
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff_gaussian_rasterization_4d
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/gaussian_rasterization_ch3
pip install submodules/gaussian_rasterization_ch9
pip install submodules/diff_gaussian_rasterization_4dch9
pip install submodules/simple-knn

# create wandb project following instruction
wandb init  
```

## Usage


### On Our Instructive Dataset

We provide a python utility to run training and testing for the instructive dataset. The utility trains the model as well as runs evaluations for masked and non-masked metrics.
The utility by default is going to run the code locally. However, this amount of experiments will likely require a cluster to finish in a reasonable time. For this case, the utility can instead run scripts on the slurm cluster. Check slurms/custom.sh as an example and specify ```--slurm_script``` parameter.
For training and testing method ```${method}``` on the instructive dataset scene ```${scene}```,

```bash
exp_group_name="vanilla"
exp_name="${scene}_${method}"

python runner.py \
    --config_file configs/custom/${method}/vanilla1.yaml \
    --group ${exp_group_name}_${scene} \
    --name ${exp_name} \
    --dataset data/custom/${scene} \
    --slurm_script slurms/custom.sh \
    --output_dir output/custom/${exp_group_name}/${scene}/${method}
```

### On All Other Datasets

For training and testing method ```${method}``` on dataset ```${dataset}```'s scene ```${scene}```,

```bash
base="${dataset}/${scene}/${method}"
name="vanilla1"
variant="${base}/${name%?}1"
output_path="./output/${base}"

python main.py fit \
    --config configs/${variant}.yaml \
    --output ${output_path} \
    --name "${base##*/}_$name" 

python main.py test \
    --config configs/${variant}.yaml \
    --ckpt_path  last \
    --output ${output_path} \
    --name "${base##*/}_$name" 
```

### Citation

If you find our repository useful, please consider giving it a star ⭐ and citing our paper:

```
@misc{liang2024monocular,
  title        ={Monocular Dynamic Gaussian Splatting: Fast, Brittle, and Scene Complexity Rules},
  author       ={Yiqing Liang and Mikhail Okunev and Mikaela Angelina Uy and Runfeng Li and Leonidas Guibas and James Tompkin and Adam W. Harley},
  year         ={2024},
  eprint       ={2412.04457},
  archivePrefix={arXiv},
  primaryClass ={cs.CV}
}
```
