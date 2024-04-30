# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from collections import Counter
import os
from functions import *
from pprint import pprint
from utils import IntentsAndSlots, collate_fn, load_data, Lang
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from model import JointIntentSlotsBert

if __name__ == "__main__":
    
    # Load the data
    train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    dev_raw = load_data(os.path.join('dataset','ATIS','dev.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader = DataLoader(train_raw, batch_size=128, collate_fn=lambda x: collate_fn(x, tokenizer), shuffle=True)
    dev_loader = DataLoader(dev_raw, batch_size=64, collate_fn=lambda x: collate_fn(x, tokenizer))
    test_loader = DataLoader(test_raw, batch_size=64, collate_fn=lambda x: collate_fn(x, tokenizer))

    model = JointIntentSlotsBert(out_slot=128, out_int=22, dropout=0.1)
    
