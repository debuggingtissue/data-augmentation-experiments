from utils import path_utils
import random
import os
from shutil import copyfile


class EnsembleValidationDataManager:

    def __init__(self, dataset_path, validation_images_from_each_class):
        self.dataset_path = dataset_path
        self.validation_images_from_each_class = validation_images_from_each_class
        self.ensemble_dataset_validation_path = self.generate_ensemble_validation_data()

    def generate_ensemble_validation_data(self):
        spop_true_origin_directory_path = self.dataset_path + "/SPOP_true"
        spop_true_origin_files = path_utils.get_all_files_in_directory(spop_true_origin_directory_path)
        spop_false_origin_directory_path = self.dataset_path + "/SPOP_false"
        spop_false_origin_files = path_utils.get_all_files_in_directory(spop_false_origin_directory_path)

        print(spop_true_origin_files)
        print(spop_false_origin_files)

        random.shuffle(spop_true_origin_files)
        random.shuffle(spop_false_origin_files)

        # Pop desired amount of image files to train and val arrays
        spop_true_ensemble_validation_set = []
        spop_false_ensemble_validation_set = []
        for draw in range(self.validation_images_from_each_class):
            spop_true_image_path_drawn = spop_true_origin_files.pop()
            spop_true_ensemble_validation_set.append(spop_true_image_path_drawn)

            spop_false_image_path_drawn = spop_false_origin_files.pop()
            spop_false_ensemble_validation_set.append(spop_false_image_path_drawn)

        root_validation_data_path = "ensemble_validation_data"
        os.mkdir(root_validation_data_path)

        spop_true_validation_data_path = f"{root_validation_data_path}/SPOP_true"
        os.mkdir(spop_true_validation_data_path)

        spop_false_validation_data_path = f"{root_validation_data_path}/SPOP_false"
        os.mkdir(spop_false_validation_data_path)

        for spop_true_ensemble_validation_image_path in spop_true_ensemble_validation_set:
            print(spop_true_validation_data_path)
            print(spop_true_ensemble_validation_image_path)
            copyfile(spop_true_origin_directory_path + "/" + spop_true_ensemble_validation_image_path,
                     spop_true_validation_data_path + "/" +
                     spop_true_ensemble_validation_image_path)

        for spop_false_ensemble_validation_image_path in spop_false_ensemble_validation_set:
            copyfile(spop_false_origin_directory_path + "/" + spop_false_ensemble_validation_image_path,
                     spop_false_validation_data_path + "/" +
                     spop_false_ensemble_validation_image_path)

        for x in spop_true_ensemble_validation_set: path_utils.remove_file(spop_true_origin_directory_path + "/" + x)
        for x in spop_false_ensemble_validation_set: path_utils.remove_file(spop_false_origin_directory_path + "/" + x)

        return root_validation_data_path


def dataloaders(self, dataset_path):
    return self.data_block.dataloaders(dataset_path)


def monte_carlo_draw_balanced_train_and_validation_sets(meta_ensemble_index,
                                                        ensemble_index,
                                                        train_draw_count_per_class,
                                                        validation_draw_count_per_class):
    # Fetch image files into array from path_train_images, one for each class
    spop_true_origin_files = path_utils.get_all_files_in_directory(data_paths.SPOP_TRUE_ORIGIN_DATA_PATH)
    spop_false_origin_files = path_utils.get_all_files_in_directory(data_paths.SPOP_FALSE_ORIGIN_DATA_PATH)

    # Pop desired amount of image files to train and val arrays
    spop_true_train_set = []
    spop_false_train_set = []
    for draw in range(train_draw_count_per_class):
        spop_true_image_path_drawn = spop_true_origin_files.pop()
        spop_false_image_path_drawn = spop_false_origin_files.pop()

        spop_true_train_set.append(spop_true_image_path_drawn)
        spop_false_train_set.append(spop_false_image_path_drawn)

    spop_true_validation_set = []
    spop_false_validation_set = []
    for draw in range(validation_draw_count_per_class):
        spop_true_image_path_drawn = spop_true_origin_files.pop()
        spop_false_image_path_drawn = spop_false_origin_files.pop()

        spop_true_validation_set.append(spop_true_image_path_drawn)
        spop_false_validation_set.append(spop_false_image_path_drawn)

    # print("######################################")
    # print(spop_true_train_set)
    # print(spop_false_train_set)
    # print(spop_true_validation_set)
    # print(spop_false_validation_set)

    # Create folder with meta_ensamble_index and ensamble_index in name
    splits_data_path = pathlib.Path(data_paths.ENSEMBLES_DATA_PATH)
    splits_data_path.mkdir(parents=True, exist_ok=True)

    unique_monte_carlo_split_directory = "ensemble_" + str(meta_ensemble_index) + "_model_" + str(
        ensemble_index)
    monte_carlo_split_path = splits_data_path / unique_monte_carlo_split_directory

    if monte_carlo_split_path.is_dir():
        return monte_carlo_split_path
    monte_carlo_split_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_train_path = splits_data_path / unique_monte_carlo_split_directory / constants.TRAIN
    monte_carlo_split_train_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_train_SPOP_true_path = splits_data_path / unique_monte_carlo_split_directory / constants.TRAIN / constants.SPOP_MUTATED
    monte_carlo_split_train_SPOP_true_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_train_SPOP_false_path = splits_data_path / unique_monte_carlo_split_directory / constants.TRAIN / constants.SPOP_NOT_MUTATED
    monte_carlo_split_train_SPOP_false_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_validation_path = splits_data_path / unique_monte_carlo_split_directory / constants.VALIDATION
    monte_carlo_split_validation_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_validation_SPOP_true_path = splits_data_path / unique_monte_carlo_split_directory / constants.VALIDATION / constants.SPOP_MUTATED
    monte_carlo_split_validation_SPOP_true_path.mkdir(parents=True, exist_ok=True)

    monte_carlo_split_validation_SPOP_false_path = splits_data_path / unique_monte_carlo_split_directory / constants.VALIDATION / constants.SPOP_NOT_MUTATED
    monte_carlo_split_validation_SPOP_false_path.mkdir(parents=True, exist_ok=True)

    # copy the image paths to new location
    print(len(spop_true_train_set))
    for training_image_SPOP_true_path in spop_true_train_set:
        copyfile(training_image_SPOP_true_path, str(monte_carlo_split_train_SPOP_true_path) + "/" +
                 str(training_image_SPOP_true_path.resolve()).split('/')[-1])

    print(len(spop_false_train_set))
    for training_image_SPOP_false_path in spop_false_train_set:
        copyfile(training_image_SPOP_false_path, str(monte_carlo_split_train_SPOP_false_path) + "/" +
                 str(training_image_SPOP_false_path.resolve()).split('/')[-1])

    print(len(spop_true_validation_set))
    for validation_image_SPOP_true_path in spop_true_validation_set:
        copyfile(validation_image_SPOP_true_path, str(monte_carlo_split_validation_SPOP_true_path) + "/" +
                 str(validation_image_SPOP_true_path.resolve()).split('/')[
                     -1])

    print(len(spop_false_validation_set))
    for validation_image_SPOP_false_path in spop_false_validation_set:
        copyfile(validation_image_SPOP_false_path, str(monte_carlo_split_validation_SPOP_false_path) + "/" + \
                 str(training_image_SPOP_false_path.resolve()).split('/')[
                     -1])

    return monte_carlo_split_path
