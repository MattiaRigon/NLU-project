# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from collections import Counter
import os
from functions import *
from transformers import BertTokenizer, BertModel
from pprint import pprint
from utils import IntentsAndSlots, collate_fn, load_data, Lang
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader



if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    
    # Load the data
    train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    dev_raw = load_data(os.path.join('dataset','ATIS','dev.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

    train_dataset = IntentsAndSlots(train_raw, lang=None)
    dev_dataset = IntentsAndSlots(dev_raw, lang=None)
    test_dataset = IntentsAndSlots(test_raw, lang=None)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    model = BertModel.from_pretrained("bert-base-uncased") # Download the model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer

    


    
