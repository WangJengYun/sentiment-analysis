from .base_callback import TrainerCallback
try:
    from torch.utils.tensorboard import SummaryWriter  

except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )

class TensorBoardCallback(TrainerCallback):
    def initial(self, args, state, control, **kwargs):
        self.tb_writer = SummaryWriter(log_dir = args.log_path)

    def on_train_begin(self, args, state, control, record = None, model_config = None, **kwargs):
        combined_dict = dict()

        if record is not None:
            combined_dict = {**record,**combined_dict}

        if model_config is not None:
            combined_dict = {**model_config,**combined_dict}

        self.tb_writer.add_text("trainer_config", str(combined_dict))

    def on_train_end(self, args, state, control, **kwargs):
        pass
    
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    
    def on_log(self, args, state, control, metrics, step, mode, **kwargs):
        
        log_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                log_metrics[k] = v
                self.tb_writer.add_scalar('{}/{}'.format(k,mode),v, step)

    def on_save(self, args, state, control, **kwargs):
        pass