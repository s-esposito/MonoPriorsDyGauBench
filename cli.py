from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from callbacks import WandbWatcher
from typing import Optional, List
import os
from jsonargparse import Namespace, lazy_instance

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.add_argument("--name", "-n", type=Optional[str], default=None,
            help="where to store result: ./output/something")
        parser.add_argument("--output", "-o", type=Optional[str], default="./output",
            help="where to store result: something/name")
        parser.add_argument("--logger", type=str, default="wandb")
        parser.add_argument("--project", "-p", type=Optional[str], default="base",
            help="wandb project name")
        parser.add_argument("--group", "-g", type=Optional[str], default="default",
            help="wandb group name")
        #parser.link_arguments("data.batch_size", "model.batch_size")

    # called before instantiating the trainer
    # 1. create output path
    # 2. if checkpoint is found, load checkpoint
    # 3. set logger
    # referred to https://github.com/yzslab/gaussian-splatting-lightning/blob/main/internal/cli.py
    def before_instantiate_classes(self) -> None:
        # self.config exists, but is empty!
        # anyways need to read config based on subcommand
        config = getattr(self.config, self.config.subcommand)
        assert config.name is not None, "Experiment must have a name for saving and logging!"
        assert config.output is not None, "Experiment must have a base output path!"
        
        # build output path
        output_path = os.path.join(config.output, config.name)
        os.makedirs(output_path, exist_ok=True)
        print("output path: {}".format(output_path))

        # find checkpoint if ckpt_path is set to "last" instead of exact path
        if config.ckpt_path == "last":
            config.ckpt_path = os.path.join(
                output_path, "last.ckpt"
            )
        if self.config.subcommand == "fit":
            if config.ckpt_path is None:
                assert os.path.exists(
                    os.path.join(output_path, "point_cloud")
                ) is False, ("point cloud output already exists in {}, \n"
                             "please specific a different experiment name (-n)").format(output_path)
        
        else:
            # disable logger
            config.logger = "None"
            # disable config saveing
            self.save_config_callback = None
            if config.ckpt_path is None:
                config.ckpt_path = os.path.join(
                    output_path, "last.ckpt"
                )    
        if config.ckpt_path: 
            assert (config.ckpt_path.endswith(".ckpt") and os.path.exists(config.ckpt_path)), f"ckpt_path {config.ckpt_path} is not legit" 
        

        # build logger
        logger_config = Namespace(
            class_path=None,
            init_args=Namespace(
                save_dir=output_path,
            ),
        )


        if config.logger == "tensorboard":
            logger_config.class_path = "lightning.pytorch.loggers.TensorBoardLogger"
        elif config.logger == "wandb":
            logger_config.class_path = "lightning.pytorch.loggers.WandbLogger"
            #wandb_name = config.name
            #if config.version is not None:
            #    wandb_name = "{}_{}".format(wandb_name, config.version)
            setattr(logger_config.init_args, "name", config.name)
            setattr(logger_config.init_args, "project", config.project)
            setattr(logger_config.init_args, "group", config.group)
            #setattr(logger_config.init_args, "log_model", "all")
            if config.trainer.callbacks is None:
                config.trainer.callbacks = []
            assert isinstance(config.trainer.callbacks, list)
            config.trainer.callbacks += [lazy_instance(WandbWatcher)]

        
        elif config.logger == "none" or config.logger == "None" or config.logger == "false" or config.logger == "False":
            logger_config = False
        else:
            assert False, "Undefined logger!"
            #logger_config.class_path = config.logger

        config.trainer.logger = logger_config        
    
    #def instantiate_classes(self) -> None:
    #    super().instantiate_classes()
    #    self.trainer.logger.watch(self.model, log="all")




