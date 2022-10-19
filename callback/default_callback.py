from .base_callback import TrainerCallback

class DefaultFlowCallback(TrainerCallback):
    def initial(self, args, state, control, **kwargs):
        pass

    def on_step_begin(self, args, state, control, log_args, **kwargs):
        
        if log_args.is_log == True:
            control.should_log = True
        
        if state.is_validation & (state.num_train_epochs %  log_args.evaluation_period == 0):
            control.should_evaluate = True    

        if (state.num_train_epochs %  log_args.save_period == 0) and control.should_evaluate :
            control.should_save = True

        return control

    def on_step_end(self, args, state, control, **kwargs):
        
        control.should_evaluate = False
        control.should_save = False
        control.should_save_best_model = False
        
        return control

