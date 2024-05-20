import torch.nn as nn
from transformers import BertTokenizer, BertModel

class JointIntentSlotsBert(nn.Module):

    def __init__(self,out_slot,out_int,dropout=0.1,):

        self.device = 'cuda:0'

        super(JointIntentSlotsBert, self).__init__()

        # Carica il modello BERT pre-addestrato e il tokenizer
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.slot_out = nn.Linear(self.bert_model.config.hidden_size, out_slot)  # Update from 200 to 400
        self.intent_out = nn.Linear(self.bert_model.config.hidden_size, out_int)  # Update from 200 to 400
        # Dropout layer How/Where do we apply it?
        self.dropout = nn.Dropout(dropout)


    def forward(self,inputs):



        bert_output = self.bert_model(**inputs)

        # Estrai gli embeddings dall'ultimo layer di BERT
        #bert_embeddings = bert_output.last_hidden_state

        # slot_output = self.slot_out(bert_embeddings[:, 0, :])  # Assuming you're using the CLS token for classification
        # intent_output = self.intent_out(bert_embeddings[:, 0, :])  # Assuming you're using the CLS token for classification

        intent_output = bert_output.last_hidden_state[:,0,:]#bert_output[1]  # [CLS]
        sequence_output = bert_output.last_hidden_state   #[0]

        intent_output = self.intent_out(intent_output)
        slot_output = self.slot_out(sequence_output)

        ## SOFTMAX ? 
        slot_output = slot_output.permute(0,2,1)

        return slot_output, intent_output 

        