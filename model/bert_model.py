from typing import List, Optional, Tuple, Union
import torch 
from torch import nn 

from .basic_model import BaseModel
from transformers import BertModel,BertPreTrainedModel

from .model_output import SequenceClassifierOutput

class BertForSequenceClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        # pooled_output = self.LayerNorm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits
        )
        
if __name__ == '__main__':
    
    # load model
    pretrained_model_name_or_path = 'C:\\Users\\cloudy822\\Desktop\\sentiment_analysis\\model\\pretrained_model\\ckip-bert-base-chinese'
    num_labels = 2
    model = BertForSequenceClassifier.from_pretrained(
        pretrained_model_name_or_path = pretrained_model_name_or_path,
        num_labels = num_labels
    )

    # model ouptut
    input_ids = torch.tensor(
        [[101,2595, 5543,  679, 7097,  117, 7705, 1240, 4676,  679, 3118, 2898],
         [101,2595, 5543,  679, 7097,  117, 7705, 1240, 4676,  679, 3118, 2898],
         [101,2595, 5543,  679, 7097,  117, 7705, 1240, 4676,  679, 3118, 2898]]
    )
    labels = torch.tensor([[0,1,1]])
    output = model(input_ids = input_ids,return_dict = True)