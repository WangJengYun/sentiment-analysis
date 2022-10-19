import torch 
import logging
from pathlib import Path
from functools import partial, update_wrapper
import trainer.metric as metric_module
import trainer.loss as loss_module
import trainer.lr_scheduler as lr_scheduler_module
from utils.converfunc import Convert2dict
from utils.exceptions import WrongArgumentsError

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self, training_config):

        # self.training_config = Convert2dict(config, 'training_config')
        self.training_config = training_config
        
        self.empty_optimizer = self.init_object(
            self.training_config['optimizer'], 
            default_module = torch.optim
        )
        self.empty_lr_scheduler = self.init_object(
            self.training_config['lr_scheduler'], 
            customized_module = lr_scheduler_module,
            num_epoch = self.training_config['batch_size'],
            num_training_steps_per_epoch = self.training_config['num_epoch']
        )
        self.criterion = self.init_object(
            self.training_config['loss'],
            customized_module = loss_module,
            num_labels = self.training_config['num_labels']
        )

        self.metrics = self.init_object(
            self.training_config['metric'],
            customized_module = metric_module,
            num_labels = self.training_config['num_labels']
        )

        self.device = self._detect_device(self.training_config['device'])

    def init_object(
        self, 
        module_config,
        default_module=None,
        customized_module=None,
        **arg
        ):
        
        module_name = None
        input_args = None
        if isinstance(module_config, dict):
            module_name,input_args = list(module_config.items())[0]
        else:
            module_name = module_config
        
        module_args = None
        if bool(arg) and bool(input_args):
            module_args = arg
            module_args.update(input_args)
        elif bool(arg):
            module_args = arg
        else:
            module_args = input_args

        output_module = None
        if isinstance(module_name, list):
            output_module = [
             self.get_object(name, default_module, customized_module) for name in module_name
            ]
            if module_args:
                output_module = [
                    update_wrapper(partial(module, **module_args), module) for module in output_module
                ] 
        else:
            output_module = self.get_object(module_name, default_module, customized_module)
            if module_args:
                output_module = partial(output_module, **module_args)

        return output_module

    def get_object(self, module_name, default_module, customized_module):
        output_module = None

        if (customized_module is not None) and (hasattr(customized_module, module_name)):
            output_module = getattr(customized_module, module_name)
        elif (default_module is not None) and (hasattr(default_module, module_name)):
            output_module = getattr(default_module, module_name)
        else:    
            WrongArgumentsError('{} is not found in module_name'.format(module_name))

        return output_module

    def _save_pretrained(self, model, saving_path):

        logger.info("Saving current best: model_best.pth ...")
        model.save_pretrained(saving_path)        
    
    def _save_checkpoint(self, epoch, model, metrics, is_best = False):
        save_dict = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'monitor_metric': metrics,
            'config': self.training_config
        }
        
        checkpoint_folder = self.config.saving_folder / 'checkpoint' 
        checkpoint_folder.mkdir(parents = True, exist_ok = True)
        
        if is_best:
            logger.info("Saving current best: model_best.pth ...")
            
            best_filename = self.config.saving_folder /'best_model.pth'
            torch.save(save_dict, best_filename)
        else:
            filename = 'checkpoint-epoch{}.pth'.format(epoch)
            logger.info("Saving checkpoint: {} ...".format(filename))
        
            torch.save(save_dict, checkpoint_folder / filename)

    def _detect_device(self,device):
        device_name, device_num = device.split(':')
        if device_name == 'cuda':
            device_num =  int(device_num) if device_num else 0
            if torch.cuda.is_available() and (device_num <= (torch.cuda.device_count() - 1 )):
                device = device
            else:
                device = 'cpu'
        return device

if __name__ == '__main__':
    import os 
    os.chdir('C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\')
    import logging
    from configuration import ConfigParser
    from data_loader import SentimentDataLoader
    from model import BertForSequenceClassifier
    from trainer import Trainer

    config = ConfigParser('config.ini') 
    output = BaseTrainer(config.training_config)