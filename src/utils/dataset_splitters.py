from utils import path_utils, constants
import random
import pathlib
from shutil import copyfile


# def split_into_train_validation_test_sets(full_dataset, train_proportion, validation_proportion, test_proportion):
#     train_and_validation_set_count = int(len(full_dataset) * (train_proportion + validation_proportion))
#     test_set_count = int(len(full_dataset) * test_proportion)
#
#     train_and_validation_set, test_set = torch.utils.data.random_split(full_dataset, [train_and_validation_set_count
#         , test_set_count])
#
#     train_set_count = int(len(train_and_validation_set) * train_proportion / (train_proportion + validation_proportion))
#     validation_set_count = int(
#         len(train_and_validation_set) * validation_proportion / (train_proportion + validation_proportion))
#
#     train_set, validation_set = torch.utils.data.random_split(train_and_validation_set,
#                                                               [train_set_count, validation_set_count])
#     return train_set, validation_set, test_set
#
#
# def split_set_into_examples_and_labels(dataset):
#     all_examples = []
#     all_labels = []
#     for data_tuple in dataset:
#         all_examples.append(data_tuple[0])
#         all_labels.append(data_tuple[1])
#
#     return all_examples, all_labels


def monte_carlo_draw_balanced_train_and_validation_sets(meta_ensemble_index,
                                                        ensemble_index,
                                                        train_draw_count_per_class=1,
                                                        validation_draw_count_per_class=1):
    # Fetch image files into array from path_train_images, one for each class
    spop_true_origin_files = path_utils.get_all_files_in_directory("dataset/SPOP_true")
    spop_false_origin_files = path_utils.get_all_files_in_directory("dataset/SPOP_false")

    random.shuffle(spop_true_origin_files)
    random.shuffle(spop_false_origin_files)

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

    print("######################################")
    print(spop_true_train_set)
    print(spop_false_train_set)
    print(spop_true_validation_set)
    print(spop_false_validation_set)

    # Create folder with meta_ensamble_index and ensamble_index in name
    splits_data_path = pathlib.Path("ensemble_datasets")
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

        copyfile("dataset/SPOP_true/" + training_image_SPOP_true_path, str(monte_carlo_split_train_SPOP_true_path) + "/" + training_image_SPOP_true_path)

    print(len(spop_false_train_set))
    for training_image_SPOP_false_path in spop_false_train_set:
        copyfile("dataset/SPOP_false/" + training_image_SPOP_false_path, str(monte_carlo_split_train_SPOP_false_path) + "/" + training_image_SPOP_false_path)

    print(len(spop_true_validation_set))
    for validation_image_SPOP_true_path in spop_true_validation_set:
        copyfile("dataset/SPOP_true/" + validation_image_SPOP_true_path, str(monte_carlo_split_validation_SPOP_true_path) + "/" + validation_image_SPOP_true_path)

    print(len(spop_false_validation_set))
    for validation_image_SPOP_false_path in spop_false_validation_set:
        copyfile("dataset/SPOP_false/" + validation_image_SPOP_false_path, str(monte_carlo_split_validation_SPOP_false_path) + "/" + training_image_SPOP_false_path)

    return monte_carlo_split_path
