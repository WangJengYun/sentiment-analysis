import importlib
from .default_callback import DefaultFlowCallback

def is_mlflow__available():
    return importlib.util.find_spec('mlflow') is not None 

def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or\
           importlib.util.find_spec("tensorboardX") is not None or\
           importlib.util.find_spec("torch.utils.tensorboard") is not None

def get_available_callback(callback_args):
    callbacks = []
    callbacks.append(DefaultFlowCallback)

    if is_tensorboard_available() and callback_args.tensorboard:
        from .tensorboard_callback import TensorBoardCallback
        callbacks.append(TensorBoardCallback)

    if is_mlflow__available() and callback_args.mlflow :
        from .mlflow_callback import MLflowCallback
        callbacks.append(MLflowCallback)

    if callback_args.earlystopping:
        from .earlystopping_callback import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback)

    return callbacks
