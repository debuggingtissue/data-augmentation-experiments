# from utils import fastai_cluster_plots_utils, transform_definitions_generator, data_block_manager, ensemble_validation_data_manager, ensemble_manager
# from fastai.vision.all import *
#
# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import ntpath
import os.path
from utils import ensemble_validation_data_manager, transform_definitions_generator, data_block_manager, \
    ensemble_manager
import shutil, errno
from pathlib import Path

ntpath.basename("a/b/c")


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else:
            raise


def train_spop_classifier(dataset_path):
    dataset_path = Path(dataset_path)
    # current_path = Path.cwd()
    # dataset_path = current_path / dataset_name
    # files = get_image_files(dataset_path)
    spop_data_block = DataBlock(blocks=(ImageBlock, CategoryBlock),
                                get_items=get_image_files,
                                splitter=RandomSplitter(),
                                get_y=parent_label,
                                item_tfms=Resize(460),
                                batch_tfms=aug_transforms(size=224, min_scale=0.75))

    # print(spop_data_block.summary(dataset_path))
    dls = spop_data_block.dataloaders(dataset_path)
    # # # dls.show_batch(nrows=4, ncols=3)
    # print(dls.get_idxs)
    # # plt.show()
    # learn = cnn_learner(dls, resnet34, pretrained=False, metrics=error_rate)
    # learn.fine_tune(2)

    learn = cnn_learner(dls, resnet50, pretrained=True, metrics=error_rate)
    learn.fit_one_cycle(20, 3e-3)
    loss_plot = get_plot_loss(learn.recorder)
    plt.savefig(f'first_loss_plot_{path_leaf(dataset_path)}.png')
    learn.unfreeze()
    learn.lr_find(show_plot=False)
    lr_plot = get_plot_lr_find(learn.recorder)
    plt.savefig(f'lr_find_{path_leaf(dataset_path)}.png')
    learn.fit_one_cycle(100, lr_max=6e-3)
    loss_plot = get_plot_loss(learn.recorder)
    plt.savefig(f'second_loss_plot_{path_leaf(dataset_path)}.png')
    learn.export()


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def remove_old_datasets():
    shutil.rmtree("dataset")
    shutil.rmtree("ensemble_validation_data")
    shutil.rmtree("ensemble_datasets")


def create_copy_of_source_dataset():
    dataset_origin_path_source = "origin/dataset_origin"
    dataset_path = "dataset"
    copyanything(dataset_origin_path_source, dataset_path)
    return dataset_path


if __name__ == '__main__':
    run_data_augmentation_experiment_1()


def run_data_augmentation_experiment():
    remove_old_datasets()
    source_copy_dataset_path = create_copy_of_source_dataset()

    data_augmentation_transforms = transform_definitions_generator.generate_baseline_item_and_batch_transforms()
    data_block_manager = data_block_manager.DataBlockManager(data_augmentation_transforms)

    models = create_arbitary_amount_models(models_to_train=5, data_block_manager)



def create_arbitary_amount_models(models_to_train, data_block_manager):
    ensemble_count = 1

    # extract x images from positive and negative
    models = ensemble_manager.generate_ensembles(ensemble_count,
                                                 models_to_train,
                                                 data_block_manager)
    return models


def run_monte_carlo_experiment():
    # train_spop_imbalance_classifier("../dataset_origin")
    # train_spop_classifier("../dataset_46_balanced")

    remove_old_datasets()
    source_copy_dataset_path = create_copy_of_source_dataset()

    ensemble_validation_data_manager = ensemble_validation_data_manager.EnsembleValidationDataManager(
        source_copy_dataset_path, validation_images_from_each_class=10)
    train_draw_count_per_class = 1,
    validation_draw_count_per_class = 1
    data_augmentation_transforms = transform_definitions_generator.generate_baseline_item_and_batch_transforms()
    data_block_manager = data_block_manager.DataBlockManager(data_augmentation_transforms)

    ensemble_count = 1
    weak_learner_count_in_each_ensemble = 3  # must be odd number to avoid tie

    # extract x images from positive and negative
    ensemble = ensemble_manager.generate_ensembles(ensemble_count,
                                                   weak_learner_count_in_each_ensemble,
                                                   data_block_manager)

    # def run_demo():
    # path = untar_data(URLs.MNIST_SAMPLE)
    # print(path.ls())
    # files = get_image_files(path / "train")
    # files = None
# pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
#                  get_items=get_image_files,
#                  splitter=RandomSplitter(),
#                  get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
#                  item_tfms=Resize(460),
#                  batch_tfms=aug_transforms(size=224))
# dls = pets.dataloaders(untar_data(URLs.PETS) / "images")
