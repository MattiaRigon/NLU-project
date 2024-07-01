import torch.nn as nn
from transformers import BertTokenizer, BertModel

class SABert(nn.Module):

    def __init__(self,config,out_slot,dropout=0.1,):

        super(SABert, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased',config=config)
        self.slot_out = nn.Linear(config.hidden_size, out_slot)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (dict): Input dictionary containing the inputs for the model.

        Returns:
            torch.Tensor: Output tensor from the model.

        """
        bert_output = self.bert_model(**inputs)
        sequence_output = bert_output.last_hidden_state  
        slot_output = self.slot_out(sequence_output)
        slot_output = slot_output.permute(0, 2, 1)

        return slot_output

        