import os.path
from os import path

class RunArguments():
    def __init__(self,
                 train_examples_draw_count_per_class,
                 ensemble_count,
                 weak_learner_count_in_each_ensemble,
                 learning_rate_first_one_cycle,
                 epochs_first_one_cycle,
                 batch_size,
                 save_plots=True):
        self.weak_learner_count_in_each_ensemble = weak_learner_count_in_each_ensemble
        self.learning_rate_first_one_cycle = learning_rate_first_one_cycle
        self.epochs_first_one_cycle = epochs_first_one_cycle
        self.ensemble_count = ensemble_count
        self.train_examples_draw_count_per_class = train_examples_draw_count_per_class
        self.batch_size = batch_size
        self.save_plots = save_plots


    def plot_save_path(self):
        results_directory = "results"
        if path.isdir(results_directory) is False:
            os.mkdir(results_directory)

        experiment_dictionary_name = f"ted={self.train_examples_draw_count_per_class}###bs={self.batch_size}###lr_oc1={self.learning_rate_first_one_cycle}###ep_oc1={self.epochs_first_one_cycle}"
        experiment_dictionary = results_directory + "/" + experiment_dictionary_name
        if path.isdir(experiment_dictionary) is False:
            os.mkdir(experiment_dictionary)

        return experiment_dictionary
