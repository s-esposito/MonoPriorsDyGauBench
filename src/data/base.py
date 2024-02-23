from lightning import LightningDataModule

class MyDataModuleBaseClass(LightningDataModule):
    def __init__(self):
        super().__init__()