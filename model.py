import torch
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertPreTrainedModel
import torch.nn.functional as F

class BertForCSC(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCSC, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()
    
    def load_liner(self):
        self.cls.predictions.decoder.weight.data.copy_(self.bert.embeddings.word_embeddings.weight.data)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)[0]
        prediction_scores = self.cls(output)
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            _, prediction_scores = torch.max(prediction_scores, dim=-1)
            return prediction_scores