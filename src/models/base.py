from lightning import LightningModule

class MyModelBaseClass(LightningModule):
    def __init__(self):
        super().__init__()