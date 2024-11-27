# Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps

This repository is the official PyTorch implementation of the paper:

&nbsp;&nbsp;[**Monocular Dynamic Gaussian Splatting is Fast and Brittle but Smooth Motion Helps**](https://lynl7130.github.io/MonoDyGauBench.github.io/)  
[Yiqing Liang](https://lynl7130.github.io), [Mikhail Okunev](https://mmehas.github.io/), [Mikaela Angelina Uy](https://mikacuy.github.io/)†‡, [Runfeng Li](https://www.linkedin.com/in/runfeng-l-a41b6a204/), [Leonidas Guibas](https://profiles.stanford.edu/leonidas-guibas)‡, [James Tompkin](https://jamestompkin.com/), [Adam W Harley](https://adamharley.com/)‡  

<img width="12%"  text-align="center" margin="auto" src=images/brownlogo.svg> &nbsp;&nbsp;
†<img width="20%"  text-align="center" margin="auto" src=images/nvidia_logo.png>  &nbsp;&nbsp;
‡<img width="8%"  text-align="center" margin="auto" src=images/stanfordlogo.png>

&nbsp;&nbsp;&nbsp;[Paper](https://lynl7130.github.io/data/DyGauBench_tmp.pdf)

## Installation

1. check ```slurms/install.sh```

2. initialize wandb

After activating conda environment
```
wandb init

paste API key and create first project following instruction
```

3. PytorchLightning version was tested on 2.2.1

## Usage


Default usage following PytorchLightning:

```python main.py [before] [subcommand] [after]```

As do not want to customize arguments by subcommand (fit/test), pass the config after the subcommand

```python main.py [subcommand] --config path/to/config.yaml```

Note: when building extension using pip install, have to run on the cluster where the pytorch is installed.
After building, can switch to different cluster.