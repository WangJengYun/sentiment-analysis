import logging
from utils.exceptions import ValueArgumentsError

logger = logging.getLogger(__name__)

class TrainerCallback:

    def initial(self, args, state, control, **kwargs):
        pass 

    def on_train_begin(self, args, state, control, **kwargs):
        pass 

    def on_train_end(self, args, state, control, **kwargs):
        pass
    
    def on_step_begin(self, args, state, control, **kwargs):
        pass 
    
    def on_step_end(self, args, state, control, **kwargs):
        pass     
    
    def on_evaluate(self, args, state, control, **kwargs):
        pass
    
    def on_log(self, args, state, control, **kwargs):
        pass

    def on_save(self, args, state, control, **kwargs):
        pass


class CallbackHandler(TrainerCallback):
    
    def __init__(self, callbacks):

        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__

        if cb_class in [c.__class__ for c in self.callbacks]:
            ValueArgumentsError(
                'You are adding a callback to the callbacks of this trainer, but there is already one. The current'
                + 'list of callbacks is:\n'
                + self.callback_list
            )
        self.callbacks.append(cb)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)
    
    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
        else:
            self.callbacks.remove(callback)

    def call_event(self, event, args, state, control, **kwargs):

        for callback in self.callbacks:
            
            result = getattr(callback, event)(args, state, control, **kwargs)

            if result is not None:
                control = result

        return control

    def initial(self, args, state, control, **kwargs):
        failed_callback = []
        
        for callback in self.callbacks:
            try:            
                result = getattr(callback, 'initial')(args, state, control, **kwargs)
                
                if result is not None:
                    control = result    
                logger.info('Complete to load the {} successfully'.format(callback.__class__.__name__))
            except Exception as e:
                failed_callback.append(callback)
                logger.warning('Fail to load the {} ...'.format(callback.__class__.__name__))
                logger.warning(e)
        
        if failed_callback:
            for cb in failed_callback:
                self.remove_callback(cb)
            logger.warning('Remove the list of callbacks :{}'.format("\n".join(cb.__class__.__name__ for cb in failed_callback)))

        return control
    
    def initial(self, args, state, control, **kwargs):
        return self.call_event('initial', args, state, control, **kwargs) 

    def on_train_begin(self, args, state, control, **kwargs):
        return self.call_event('on_train_begin', args, state, control, **kwargs) 

    def on_train_end(self, args, state, control, **kwargs):
        return self.call_event('on_train_end', args, state, control, **kwargs)

    def on_step_begin(self, args, state, control, **kwargs):
        return self.call_event('on_step_begin', args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        return self.call_event('on_step_end', args, state, control, **kwargs)

    def on_evaluate(self, args, state, control, **kwargs):
        return self.call_event('on_evaluate', args, state, control, **kwargs)
    
    def on_log(self, args, state, control, **kwargs):
        return self.call_event('on_log', args, state, control, **kwargs)

    def on_save(self, args, state, control, **kwargs):
        return self.call_event('on_save', args, state, control, **kwargs)

if __name__ == '__main__':
    pass 