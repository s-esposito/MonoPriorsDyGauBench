from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from callbacks import WandbWatcher
from typing import Optional, List
import os
from jsonargparse import Namespace, lazy_instance
try:
    from lightning.fabric.strategies import FSDPStrategy
except:
    from lightning.pytorch.strategies import FSDPStrategy
#import lightning
#print(lightning.__version__)
#assert False

#Trainer(accelerator=”gpu”, devices=k, strategy=’ddp’, precision=16)


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
        
        '''
        seed_everything(config.seed_everything, workers=True)
        data_config = getattr(config, "data")
        data_init_args = getattr(data_config, "init_args")
        # this would also affect parent configs
        setattr(data_init_args, "seed", config.seed_everything) 
        '''
        
        
        # build output path
        output_path = os.path.join(config.output, config.name)
        os.makedirs(output_path, exist_ok=True)
        print("output path: {}".format(output_path))

        if config.model.init_args.motion_mode in ["MLP", "HexPlane"]:
            config.trainer.max_steps *= 2
        elif config.model.init_args.motion_mode in ["EffGS", "FourDim", "TRBF"]:
            pass
        else:
            assert False, f"Unknown motion mode without handling of trainer steps: {config.model.init_args.motion_mode}"

        '''
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
        '''
       
        if config.ckpt_path == "last":
            checkpoint_name = sorted(os.listdir(os.path.join(
                output_path, "checkpoints"
            ))) 
            checkpoint_name = [name for name in checkpoint_name if name.startswith("last-v")]
            if len(checkpoint_name) > 0:
                name = checkpoint_name[-1]
            else:
                name = "last.ckpt"
            config.ckpt_path = os.path.join(
                output_path, "checkpoints", name
            )

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
            setattr(logger_config.init_args, "mode", "online")
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

        #if config.trainer.inference_mode:
        #    # Set trainer.testing to True
        #    config.trainer.testing = True
        
        # Add Checkpoint Logic
        #config.trainer.callbacks += [ModelCheckpoint(
        #        dirpath=os.path.join(output_path, "ckpts"), **config.trainer.checkpoint
        #    ),]
        #assert False, config.trainer.callbacks[-1]
        #config.trainer.checkpoint.dirpath = os.path.join(output_path, "ckpts")

        # if use FSDP training strategy 
        # opt to save distributed checkpoint
        # world size can change before and after save/load!
        #if config.trainer.strategy is not None:
        #    assert config.trainer.strategy == "FSDP"
        #   strategy = FSDPStrategy(state_dict_type="sharded")
        #    config.trainer.strategy = lazy_instance(strategy)
        #config.trainer.callbacks += [GlobalStepWatcher()]
        super().before_instantiate_classes()

        if config.trainer.inference_mode or self.config.subcommand != "fit":
            self.inference_mode = True
        else:
            self.inference_mode = False
        self.output_path = output_path
    def instantiate_classes(self) -> None:
        
        super().instantiate_classes()
        
        # default dirpath is null; override to save checkpoints to root of log_dir
        os.makedirs(os.path.join(self.output_path, "checkpoints"), exist_ok=True)
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                new_cb = ModelCheckpoint(
                    monitor=cb.monitor,
                    dirpath=os.path.join(self.output_path, "checkpoints"),
                    filename=cb.filename,
                    save_top_k=cb.save_top_k,
                    mode=cb.mode,
                    save_last=cb.save_last,
                    every_n_train_steps=None
                )
        self.trainer.callbacks = [cb for cb in self.trainer.callbacks if not isinstance(cb, ModelCheckpoint)]
        self.trainer.callbacks += [new_cb]
        
        # Check if we are running test; if yes, do not let overwrite config happens
        if self.inference_mode:
            # Remove SaveConfigCallback if running test
            self.trainer.testing = True
            self.trainer.callbacks = [cb for cb in self.trainer.callbacks if not isinstance(cb, SaveConfigCallback)]

        #for cb in self.trainer.callbacks:
        #    print(cb)

        #assert False
    


