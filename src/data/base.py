from lightning import LightningDataModule

class MyDataModuleBaseClass(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()