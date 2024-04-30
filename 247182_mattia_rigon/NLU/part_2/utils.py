import torch
import torch.utils.data as data
from collections import Counter
from torch.utils.data import DataLoader
import json
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
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

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
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res
    
class IntentAndSlotsBert(data.Dataset):
    def __init__(self,bert_tokenizer ,dataset, unk='unk'):
        self.bert_tokenizer = bert_tokenizer
        self.tokenized = []
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.tokenized.append(self.bert_tokenizer(x["utterance"]))
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.bert_tokenizer.convert_ids_to_token[idx])
        slots = torch.Tensor(self.bert_tokenizer.convert_ids_to_token[idx])
        intent = self.bert_tokenizer.convert_ids_to_token[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample


def collate_fn(data,bert_tokenizer):

    # data : utteance plain 
    # qui viene applciato il bert tokenizer 
    # problema che rimane è che abbiamo label come word ma tokens come subword

    new_item = {}  

    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_utt = bert_tokenizer(new_item['utterance'],return_tensors="pt", padding=True) 
    y_slots = bert_tokenizer(new_item['slots'])['input_ids']
    intent = torch.LongTensor(new_item["intent"])
    y_lengths = [len(seq) for seq in new_item['slots']]

    src_utt = src_utt.to(device) # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt # input in model
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots 
    new_item["slots_len"] = y_lengths # input in model

    return new_item
