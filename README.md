# GaussianDiff

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