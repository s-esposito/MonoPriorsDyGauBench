# GaussianDiff

## Installation

1. check ```slurms/install.sh```

2. initialize wandb

After activating conda environment
```
wandb init

paste API key and create first project following instruction
```


## Usage


Default usage following PytorchLightning:

```python main.py [before] [subcommand] [after]```

As do not want to customize arguments by subcommand (fit/test), pass the config after the subcommand

```python main.py [subcommand] --config path/to/config.yaml```
