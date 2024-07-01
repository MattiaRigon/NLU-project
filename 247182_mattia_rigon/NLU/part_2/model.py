import torch.nn as nn
from transformers import BertTokenizer, BertModel

class JointIntentSlotsBert(nn.Module):

    def __init__(self,config,out_slot,out_int,dropout=0.1,):

        super(JointIntentSlotsBert, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.slot_out = nn.Linear(config.hidden_size, out_slot)  
        self.intent_out = nn.Linear(config.hidden_size, out_int)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Input dictionary containing the inputs for the BERT model.

        Returns:
            tuple: A tuple containing the slot output and intent output.

        """
        bert_output = self.bert_model(**inputs)
        sequence_output = bert_output.last_hidden_state  
        intent_output = bert_output.pooler_output

        drop_slot = self.dropout(sequence_output)
        drop_intent = self.dropout(intent_output)
        
        slot_output = self.slot_out(drop_slot)
        intent_output = self.intent_out(drop_intent)

        slot_output = slot_output.permute(0,2,1)

        return slot_output, intent_output 

        