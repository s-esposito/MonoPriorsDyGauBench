from lightning.pytorch.callbacks import Callback


class WandbWatcher(Callback):
    def on_train_start(self, trainer, pl_module) -> None:
        print("wandb logger is watching model in all mode!")
        #assert False, "Pause"
        trainer.logger.watch(pl_module, log="all")

    def on_train_end(self, trainer, pl_module) -> None:
        print("wandb logger is unwatching model!")
        trainer.logger.experiment.unwatch(pl_module) 