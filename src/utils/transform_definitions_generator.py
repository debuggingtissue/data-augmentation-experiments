from torchvision import datasets, transforms
from fastai.vision.all import *

# For data augmentation, the 800x800px images at
# high magnification were trimmed to the centermost 512x512px ===== DONE IN DATA AQUASITION PART
# in 6 degree rotations ==== DONE IN , scaled to 256x256px, flipped and unflipped ==== DONE but WTF, then randomly cropped to 224x224 within Caffe [22]


def generate_baseline_item_and_batch_transforms():
    transforms = {
        'item': setup_aug_tfms([
            Resize(460)
        ]),
        'batch': aug_transforms(size=224, min_scale=0.75),
    }
    return transforms

