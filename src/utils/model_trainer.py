from __future__ import print_function, division

import time
import copy
from fastai.vision import *
from utils import dataset_splitters, transform_definitions_generator


def train_model_in_ensamble(ensemble_index, model_index, data_manager):
    monte_carlo_drawn_images_root_path = dataset_splitters.monte_carlo_draw_balanced_train_and_validation_sets(
        ensemble_index, model_index, 2, 1)
    transforms = transform_definitions_generator.generate_simple_fastai_transformations_for_train_and_validation_image_datasets()

    data = data_manager.generate_data_bunch_from_path(monte_carlo_drawn_images_root_path)
    print(monte_carlo_drawn_images_root_path)
    data.show_batch(3)
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(1)

    stage_1_save_name = f"ensemble_{ensemble_index}_model_{model_index}_stage_1"
    learn.save(stage_1_save_name)
    learn.unfreeze()
    # learn.lr_find(start_lr=1e-5, end_lr=1e-1)
    # learn.recorder.plot()
    plt.show()

    learn.fit_one_cycle(1, max_lr=slice(3e-5, 3e-4))
    stage_2_save_name = f"ensemble_{ensemble_index}_model_{model_index}_stage_2"
    learn.save(stage_2_save_name)
    print("Model path")
    print(learn.model_dir)
    learn.export()

    return monte_carlo_drawn_images_root_path

    # learn.load(stage_2_save_name);
    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # plt.show()


def train_model_in_ensamble(ensemble_index, model_index, dataloaders, dataset_path):
    # dataset_path = Path(dataset_path)
    # current_path = Path.cwd()
    # dataset_path = current_path / dataset_name
    # files = get_image_files(dataset_path)

    # print(spop_data_block.summary(dataset_path))
    # # # dls.show_batch(nrows=4, ncols=3)
    # print(dls.get_idxs)
    # # plt.show()
    # learn = cnn_learner(dls, resnet34, pretrained=False, metrics=error_rate)
    # learn.fine_tune(2)

    learn = cnn_learner(dataloaders, resnet50, pretrained=True, metrics=error_rate)
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
