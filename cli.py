from lightning.pytorch.cli import LightningCLI
from typing import Optional, List

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", "-n", type=Optional[str], default=None,
            help="where to store result: ./output/something")
        #parser.link_arguments("data.batch_size", "model.batch_size")

