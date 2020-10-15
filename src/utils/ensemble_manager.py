from utils import dataset_splitters, data_block_manager, model_trainer
import matplotlib.pyplot as plt


def generate_ensembles(ensemble_count, weak_learner_count_in_each_ensemble, data_block_manager):
    # data_sets, output_classes_count, dataset_sizes, device
    ensemble_dictionary = {}
    for ensemble_index in range(ensemble_count):
        ensemble_dictionary[f"ensemble_{ensemble_index}"] = []
        for weak_learner_index in range(weak_learner_count_in_each_ensemble):
            monte_carlo_drawn_images_root_path = dataset_splitters.create_monte_carlo_balanced_dataset_draw(
                ensemble_index, weak_learner_index, train_draw_count_per_class=36)
            print(monte_carlo_drawn_images_root_path)


            dataloaders = data_block_manager.dataloaders(str(monte_carlo_drawn_images_root_path), 10)

            weak_learner_in_ensemble_save_name = model_trainer.train_model_in_ensemble(ensemble_index,
                                                                                       weak_learner_index,
                                                                                       dataloaders,
                                                                                       monte_carlo_drawn_images_root_path)
            models_saved = ensemble_dictionary[f"ensemble_{ensemble_index}"]
            models_saved.append(weak_learner_in_ensemble_save_name)
    return ensemble_dictionary
