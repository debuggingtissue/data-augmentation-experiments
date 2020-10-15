from fastai.vision.all import *
from pathlib import Path
import matplotlib.pyplot as plt


class DataBlockManager:

    def __init__(self, transforms):
        self.transforms = transforms
        self.data_block = self.generate_data_block()

    def generate_data_block(self):
        data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                               get_items=get_image_files,
                               splitter=RandomSplitter(),
                               get_y=parent_label,
                               item_tfms=self.transforms["item"],
                               batch_tfms=self.transforms["batch"])

        return data_block

    def dataloaders(self, dataset_path, batch_size):
        dls = self.data_block.dataloaders(Path(dataset_path), bs=batch_size)
        return dls
