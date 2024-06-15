# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from collections import Counter
import os
from functions import *
from pprint import pprint
from utils import PAD_TOKEN,Slots, collate_fn, load_data, Lang
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertConfig
import torch.optim as optim
from tqdm import tqdm
from model import SABert
import copy
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    
    # Load the data
    device = 'cuda:0'
    tmp_train_raw = load_data(os.path.join(LOCAL_PATH,'dataset','laptop14_train.txt'))
    test_raw = load_data(os.path.join(LOCAL_PATH,'dataset','laptop14_test.txt'))

    portion = 0.10
    slots = [x['slots'] for x in tmp_train_raw]

    X_train, X_dev, y_train, y_dev = train_test_split(tmp_train_raw,slots, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        )
    train_raw = X_train
    dev_raw = X_dev

    words = sum([x['sentence'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))

    lang = Lang(words, slots, cutoff=0)

    train_dataset = Slots(train_raw, lang)
    dev_dataset = Slots(dev_raw, lang)
    test_dataset = Slots(test_raw, lang)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=lambda x: collate_fn(x), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))

    lr = 0.00005 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    vocab_len = len(lang.word2id)
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SABert(config=config,out_slot=out_slot, dropout=0.1)
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    n_epochs = 50
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = -1
    best_model = None
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 1 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang,tokenizer)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    best_model.to(device)
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, best_model, lang,tokenizer)   
    f1_result =   results_test['total']['f']
    intent_result = intent_test['accuracy']
    print('Slot F1: ',f1_result)
    print('Intent Accuracy:', intent_result)

    configurations = f'LR = {lr}\nhid_size = {config.hidden_size}\n\noptimizer={str(type(optimizer))}\nmodel={str(type(model))}\n'
    results_txt = f'{configurations}Intent Accuracy:  {intent_result}\nSlot F1: {f1_result}\nEpochs: {sampled_epochs[-1]}/{n_epochs} ' 

    save_model_incrementally(best_model,sampled_epochs,losses_train,losses_dev,results_txt)