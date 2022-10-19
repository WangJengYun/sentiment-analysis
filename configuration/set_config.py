import json
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

@dataclass
class TrainingState:
    
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    num_train_epochs: int = 0
    is_validation:bool = False
    best_metric: Optional[float] = 0
    best_model_checkpoint: Optional[str] = 'model_best'
    
    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
    
@dataclass
class TrainingControl:
        
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_save_best_model = False
    should_evaluate: bool = False
    should_log: bool = False    

    def save_to_json(self, json_path: str):
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)
    
    def _new_training(self):
        self.should_training_stop = False

    def _new_epoch(self):
        self.should_epoch_stop = False

    def _new_step(self):
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False