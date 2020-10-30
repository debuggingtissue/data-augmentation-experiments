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


