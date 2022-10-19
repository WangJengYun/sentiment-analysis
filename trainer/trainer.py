import torch 
import logging
import numpy as np 
from tqdm import tqdm

from utils.record import MetricRecord
from utils.converfunc import Convert2object

from trainer.basic_trainer import BaseTrainer
from callback import get_available_callback, CallbackHandler

logger = logging.getLogger(__name__)

class Trainer(BaseTrainer):
    def __init__(
        self,
        config, 
        model, 
        train_dataloader, 
        valid_dataloader=None
    ):
        super().__init__(config.training_config)
        self.args = Convert2object(config.training_config)
        self.log_args = Convert2object(config.log_config)
        
        self.record_list = {}
        self.state = config.training_state
        self.control = config.training_control
        self.model_config = config.model_config
    
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.state.max_steps = len(self.train_dataloader)
        self.state.is_validation = self.valid_dataloader is not None
        
        trainable_params = self.get_trainable_params(model)
        self.optimizer = self.empty_optimizer(trainable_params)
        self.lr_scheduler = self.empty_lr_scheduler(self.optimizer)

        callback_setting = Convert2object(config.callback_config)
        callbacks = get_available_callback(callback_setting)
        self.callback_handler = CallbackHandler(callbacks)
        
        self.train_metrics = MetricRecord('loss',*[f.__name__ for f in self.metrics])
        self.val_metrics = MetricRecord('loss',*[f.__name__ for f in self.metrics])
        
        self.train_metrics.reset()
        self.val_metrics.reset()

        self.callback_handler.initial(self.args, self.state, self.control)
    
    def train(self):
        self.callback_handler.on_train_begin(self.args, self.state, self.control, 
                                            record = self.record_list, model_config = self.model_config) 
        self.state.epoch = 0
        self.state.global_step = 0
        self.state.num_train_epochs = 0
        epochs_iterator = tqdm(range(self.args.num_epoch),unit = 'epoch', position = 1)
        
        for epoch in epochs_iterator:
            
            self.state.num_train_epochs = epoch + 1 
            self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control, log_args = self.log_args)
            
            train_metrics = self._train_epoch(epoch)
            if self.control.should_log:                      
                
                metrics_info = []
                for key, value in train_metrics.items():
                    metrics_info.append('{} : {}'.format(key, np.round(value, 4)))
                logger.info('Epoch {} Training   metrics ('.format(str(epoch)) + '|'.join(metrics_info) + ')')                
                
                self.callback_handler.on_log(self.args, self.state, self.control, 
                                             metrics = train_metrics, step = self.state.num_train_epochs, mode = 'train')         
                
            if self.control.should_evaluate:
                val_metrics = self._valid_epoch(epoch)
                self.state, self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, 
                                                                             metrics = val_metrics, mode = 'val')
                if self.control.should_log:
                    metrics_info = []
                    for key, value in val_metrics.items():
                        metrics_info.append('{} : {}'.format(key, np.round(value, 4)))
                    logger.info('Epoch {} Validation metrics ('.format(str(epoch)) + '|'.join(metrics_info) + ')')

                    self.callback_handler.on_log(self.args, self.state, self.control, 
                                                 metrics = val_metrics, step = self.state.num_train_epochs, mode = 'val')                                                                             
            if self.control.should_save_best_model:    
                self._save_pretrained(self.model, self.log_args.save_pretrained_path)

            epochs_iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1, train_metrics['loss']))

    def _train_epoch(self, epoch):
        self.train_metrics.reset() 
        batchs_iterator = tqdm(self.train_dataloader,unit = 'batch', position = 0)
        
        self.model.train()
        total_logits = []
        total_target = []
        for batch_idx, (_, data, target) in enumerate(batchs_iterator):
            
            data = self._todevice(data)
            target = self._todevice(target)

            # self.optimizer.zero_grad()
            self.model.zero_grad()
            
            output = self.model(**data, return_dict = True)            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
            loss = self.criterion(output.logits, target)
            
            total_logits.append(output.logits)
            total_target.append(target)
            
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.train_metrics.update(epoch, 'loss', loss.item(), iscusum = True)
            
            batchs_iterator.set_description(
                'Train_Batch {} | mean loss: {:.5f}'.format(batch_idx + 1, self.train_metrics.get_final_mertic('loss')))
            
            self.state.global_step += 1
            self.state.epoch =  self.state.num_train_epochs + (batch_idx + 1)/self.state.max_steps
            self.callback_handler.on_log(self.args, self.state, self.control, 
                                         metrics = self.train_metrics.current_performance(), step = self.state.global_step, mode = 'train')
        
        total_logits = torch.cat(total_logits, dim = 0)
        total_target = torch.cat(total_target)
        for met_fn in self.metrics:
            self.train_metrics.update(epoch, met_fn.__name__, met_fn(total_logits, total_target), iscusum = False)
        
        return self.train_metrics.performance()   

    def _valid_epoch(self, epoch):  
        self.model.eval()
        total_logits = []
        total_target = []
        self.val_metrics.reset('loss')        
        # batchs_iterator = tqdm(self.valid_dataloader, unit = 'batch', position = 0)  
        for batch_idx, (_, data, target) in enumerate(self.valid_dataloader):
            data = self._todevice(data)
            target = self._todevice(target)      
            with torch.no_grad():
                output = self.model(**data, return_dict = True)  
                loss = self.criterion(output.logits, target)
            total_logits.append(output.logits)
            total_target.append(target)
            
            self.val_metrics.update(epoch, 'loss', loss.item(), iscusum = True)
            
            # batchs_iterator.set_description(
            #         'Valid_Batch {} | mean loss: {:.5f}'.format(batch_idx + 1, self.val_metrics.get_final_mertic('loss')))

        total_logits = torch.cat(total_logits, dim = 0)
        total_target = torch.cat(total_target)
        for met_fn in self.metrics:
            self.val_metrics.update(epoch, met_fn.__name__, met_fn(total_logits, total_target), iscusum = False)
            
        return self.val_metrics.performance()
        
    def _todevice(self, data):
        if isinstance(data,dict):
            datanames = list(data.keys())
            for dataname in datanames:
                data[dataname]= data[dataname].to(self.device)
        else:
            data = data.to(self.device)
        return data
    
    def get_trainable_params(self, model):
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':  self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        return  optimizer_grouped_parameters



            
