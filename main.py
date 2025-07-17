from cli import MyLightningCLI
from src.models import MyModelBaseClass
from src.data import MyDataModuleBaseClass

if __name__ == "__main__":
    cli = MyLightningCLI(
        MyModelBaseClass,
        MyDataModuleBaseClass,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
