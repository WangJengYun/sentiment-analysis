import numpy as np 
from .base_callback import TrainerCallback

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self):
        self.early_stopping_patience = None
        self.early_stopping_threshold = None
        self.early_stopping_patience_counter = 0

        self.monitor_strategy = None
        self.monitor_metric = None
        self.monitor_type = None
        self.monitor_metric_name = None

    def initial(self, args, state, control, **kwargs):
        self.early_stopping_patience = args.early_stopping_patience
        self.early_stopping_threshold = args.early_stopping_threshold
        self.early_stopping_patience_counter = 0

        self.monitor_strategy, self.monitor_metric = args.monitor.split(' ')
        self.monitor_type, self.monitor_metric_name = self.monitor_metric.split('_', 1)
        
        assert self.monitor_strategy in ['min','max'], \
            'Please check the strategy of monitor, it must be min or max. but you input value :{}'.format(self.monitor_strategy)
        
        self.operator = np.greater if self.monitor_strategy == 'max' else np.less

    def on_train_begin(self, args, state, control, **kwargs):

        assert state.is_validation == True, 'EarlyStoppingCallback requires evaluation of validation'
    
    def on_evaluate(self, args, state, control, metrics, mode, **kwargs):
        assert mode == self.monitor_type, 'Please check the type of metric, it must be {}'.format(self.monitor_type)
        current_metric = metrics[self.monitor_metric_name]
        print(current_metric)
        print(state.best_metric)
        print(self.operator(current_metric, state.best_metric))
        if (state.best_metric == 0) or (
            self.operator(current_metric, state.best_metric)
            and abs(current_metric - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
            state.best_metric = current_metric

        else:
            self.early_stopping_patience_counter += 1

        if self.early_stopping_patience_counter == 0 :
            control.should_save_best_model = True
        else:
            control.should_save_best_model = False
        print( control.should_save_best_model)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

        return state, control 
