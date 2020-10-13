from fastai.vision.all import *

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

        # spop_data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
        #                             get_items=get_image_files,
        #                             splitter=RandomSplitter(),
        #                             get_y=parent_label,
        #                             item_tfms=Resize(460),
        #                             batch_tfms=aug_transforms(size=224, min_scale=0.75))
        return data_block

    def dataloaders(self, dataset_path):
        return self.data_block.dataloaders(dataset_path)

