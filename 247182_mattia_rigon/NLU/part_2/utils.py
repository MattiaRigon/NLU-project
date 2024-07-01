import torch
import torch.utils.data as data
from collections import Counter
import json
from transformers import BertTokenizer


PAD_TOKEN = 0
device = 'cuda:0'


def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''

    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
 

    
class IntentsAndSlots (data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids,self.slot_ids = self.mapping_seq(self.utterances,self.slots, lang.slot2id)
        self.intent_ids  = self.mapping_lab(self.intents,lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, utterances, slots, mapper): # Map sequences to number

        slot_ids = []
        utt_ids = []
        for utterance,slots_row in zip(utterances,slots):
            tokenize_slots = [PAD_TOKEN]
            sentence_id = [101]
            for word,slot in zip(utterance.split(' '),slots_row.split(' ')):
                tokens = self.bert_tokenizer([word])['input_ids'][0]
                tokens = tokens[1:len(tokens)-1]
                sentence_id.extend(tokens)
                tokenize_slots.append(mapper[slot])
                # Add padding for each subtoken except the first one
                tokenize_slots.extend([PAD_TOKEN] * (len(tokens) -1))

            sentence_id.append(102)
            tokenize_slots.append(PAD_TOKEN)
            utt_ids.append(sentence_id)
            slot_ids.append(tokenize_slots)
        
        return utt_ids,slot_ids

def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    # compute the attention mask and token type ids
    attention_mask = torch.where(src_utt != 0, torch.tensor(1), torch.tensor(0))
    token_type_ids = torch.zeros_like(attention_mask)
    # Move the tensors to the selected device
    src_utt = src_utt.to(device) 
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)
    # Create the input for the model
    input_bert = {
        "attention_mask": attention_mask,
        "input_ids": src_utt,
        "token_type_ids" : token_type_ids
    }
    new_item["utterances"] = input_bert
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item
