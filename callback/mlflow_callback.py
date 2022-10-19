
import mlflow
import mlflow.pytorch
import logging
from .base_callback import TrainerCallback

logger = logging.getLogger(__name__)

class MLflowCallback(TrainerCallback):
    def __init__(self):
        
        self._initialized = False
        self._mlflow = mlflow
        
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
    
    def initial(self, args, state, control, **kwargs):
        
        if self._mlflow.active_run() is None:
            self._mlflow.set_tracking_uri('http://192.168.0.21:5213')
            self._mlflow.set_experiment(args.project_name)
            
            self._mlflow.start_run(run_name = args.model_name)
            self._mlflow.active_run()

        self._initialized = True

    def on_train_begin(self, args, state, control, record = None, model_config = None, **kwargs):
        
        combined_dict = dict()

        if record is not None:
            combined_dict = {**record,**combined_dict}

        if model_config is not None:
            combined_dict = {**model_config,**combined_dict}
         
        for name, value in list(combined_dict.items()):
            if len(str(value)) >= self._MAX_PARAM_VAL_LENGTH:
                del combined_dict[name]
                logger.warning('Because the length of {param} is greater than {max_length},deleting {param}'.\
                                format(param = name, max_length = self._MAX_PARAM_VAL_LENGTH))

        combined_dict_items = list(combined_dict.items())

        for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
            self._mlflow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))

        self._mlflow.log_artifact(args.log_path)

    def on_train_end(self, args, state, control, **kwargs):
        self._mlflow.end_run()
    
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    
    def on_log(self, args, state, control, metrics, step, mode, **kwargs):
        
        log_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_metrics[mode + '_' + k] = v
        
        self._mlflow.log_metrics(metrics=log_metrics, step = int(step))

    def on_save(self, args, state, control, **kwargs):
        pass