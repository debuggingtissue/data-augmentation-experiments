from __future__ import print_function, division
import matplotlib

matplotlib.use('Agg')
import time
import copy
from fastai.vision.all import *
from utils import dataset_splitters, transform_definitions_generator, fastai_cluster_plots_utils, path_utils
from utils.transform_definitions_generator import generate_baseline_item_and_batch_transforms

import matplotlib.pyplot as plt


# weak_learner_in_ensemble_save_name = model_trainer.train_model_in_ensemble(ensemble_index,
#                                                                            weak_learner_index,
#                                                                            dataloaders,
#                                                                            monte_carlo_drawn_images_root_path)


# def train_model_in_ensemble(ensemble_index, model_index, dataloaders, monte_carlo_drawn_images_root_path):
#     monte_carlo_drawn_images_root_path = dataset_splitters.monte_carlo_draw_balanced_train_and_validation_sets(
#         ensemble_index, model_index, 2, 1)
#     transforms = transform_definitions_generator.generate_simple_fastai_transformations_for_train_and_validation_image_datasets()
#
#     data = data_manager.generate_data_bunch_from_path(monte_carlo_drawn_images_root_path)
#     print(monte_carlo_drawn_images_root_path)
#     data.show_batch(3)
#     learn = cnn_learner(data, models.resnet18, metrics=error_rate)
#     learn.fit_one_cycle(1)
#
#     stage_1_save_name = f"ensemble_{ensemble_index}_model_{model_index}_stage_1"
#     learn.save(stage_1_save_name)
#     learn.unfreeze()
#     # learn.lr_find(start_lr=1e-5, end_lr=1e-1)
#     # learn.recorder.plot()
#     plt.show()
#
#     learn.fit_one_cycle(1, max_lr=slice(3e-5, 3e-4))
#     stage_2_save_name = f"ensemble_{ensemble_index}_model_{model_index}_stage_2"
#     learn.save(stage_2_save_name)
#     print("Model path")
#     print(learn.model_dir)
#     learn.export()
#
#     return monte_carlo_drawn_images_root_path

# learn.load(stage_2_save_name);
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix()
# plt.show()

def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


def train_model_in_ensemble(run_arguments, ensemble_index, model_index, dataloaders, dataset_path):
    # dataset_path = Path(dataset_path)
    # current_path = Path.cwd()
    # dataset_path = current_path / dataset_name
    # files = get_image_files(dataset_path)

    # self.learning_rate_first_one_cycle = learning_rate_first_one_cycle
    # self.epochs_first_one_cycle = epochs_first_one_cycle
    # self.train_examples_draw_count_per_class = train_examples_draw_count_per_class
    # self. = batch_size

    model_arguments = f'tes: {run_arguments.train_examples_draw_count_per_class}\n' \
                      f'lr_oc1: {run_arguments.learning_rate_first_one_cycle}\n' \
                      f'bs: {run_arguments.batch_size}'
    dataset_name = path_utils.path_leaf(dataset_path)

    # print(spop_data_block.summary(dataset_path))
    # print(dataloaders)
    # dataloaders.show_batch(nrows=4, ncols=3, show=True)
    # plt.savefig(f'sample_batch_{dataset_name}.png')
    # clear_pyplot_memory()

    # print(dls.get_idxs)
    # learn = cnn_learner(dls, resnet34, pretrained=False, metrics=error_rate)
    # learn.fine_tune(2)
    experiment_results_directory = run_arguments.plot_save_path()

    learn = cnn_learner(dataloaders, resnet18, pretrained=True, metrics=error_rate)
    # 3e-2
    learn.fit_one_cycle(run_arguments.epochs_first_one_cycle, run_arguments.learning_rate_first_one_cycle)
    if (run_arguments.save_plots):

        # plt.figure(figsize=(20, 10))
        plt.figure(figsize=(10, 5))
        learn.recorder.plot_loss()
        plt.subplots_adjust(right=0.78)
        plt.text(0.82, 0.5, model_arguments, fontsize=14, transform=plt.gcf().transFigure)

        plt.title(dataset_name)

        plt.savefig(f'{experiment_results_directory}/first_loss_plot_{dataset_name}.png')
        clear_pyplot_memory()

    # learn.unfreeze()
    # learn.lr_find(show_plot=True)
    # plt.savefig(f'lr_find_{dataset_name}.png')
    # clear_pyplot_memory()

    # learn.fit_one_cycle(100, lr_max=6e-3)
    # loss_plot = get_plot_loss(learn.recorder)
    # plt.savefig(f'second_loss_plot_{path_leaf(dataset_path)}.png')
    # learn.export()


def train_model(model, criterion, optimizer, scheduler, data_loaders, dataset_sizes, device,
                validation_accuracy_threshold, num_epochs=(222 * 122)):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        should_exit_training = False
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if ((22 * 120) < epoch < (222 * 122)) and epoch_acc > 0.6:
                    should_exit_training = True
                    break
        if should_exit_training:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
