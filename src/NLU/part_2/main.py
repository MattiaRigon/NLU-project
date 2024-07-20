# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from collections import Counter
import os
from functions import *
from pprint import pprint
from utils import PAD_TOKEN,IntentsAndSlots, collate_fn, load_data, Lang
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertConfig
import torch.optim as optim
from tqdm import tqdm
from model import JointIntentSlotsBert
import copy
import argparse
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
saved_model = None

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Test application")
    parser.add_argument('--test',type=str, help='If test enabled just evaluate the model with model.pth, otherwise train')
    args = parser.parse_args()
    # Load the data
    device = 'cuda:0'
    tmp_train_raw = load_data(os.path.join(LOCAL_PATH,'dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join(LOCAL_PATH,'dataset','ATIS','test.json'))

    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents) # like a dict with intent:frequency

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)
    # if we are testing we load the model saved
    if args.test:
        try:
            saved_model = torch.load(os.path.join(LOCAL_PATH,'bin',args.test))
        except Exception as e:
            print(f"Error occured reading the weights: {e}.")
            sys.exit(1)
        lang.word2id = saved_model['word2id']
        lang.slot2id = saved_model['slot2id']
        lang.intent2id = saved_model['intent2id']
        lang.id2word = {v:k for k, v in lang.word2id.items()}
        lang.id2slot = {v:k for k, v in lang.slot2id.items()}
        lang.id2intent = {v:k for k, v in lang.intent2id.items()}
    # Create the datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)
    # Instantiate the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=lambda x: collate_fn(x), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))
    # Define the model hyperparameters
    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = JointIntentSlotsBert(config=config,out_slot=out_slot, out_int=out_int, dropout=0.4)
    model.to(device)
    model.apply(init_weights)
    # Define the optimizer and the loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    n_epochs = 50
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    accuracy_history = []
    best_f1 = -1
    best_model = None
    # If we are testing we evaluate the model
    if args.test:
        model.load_state_dict(saved_model['model'])
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang,tokenizer)
        f1_result =   results_test['total']['f']
        intent_result = intent_test['accuracy']
        print('Slot F1: ',f1_result)
        print('Intent Accuracy:', intent_result)
    # Otherwise we train the model
    else:
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model, clip=clip)
            if x % 1 == 0: # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang,tokenizer)
                accuracy_history.append(intent_res['accuracy'])
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
        # Save the model
        saving_model = {"epoch": sampled_epochs[-1], 
                    "model": best_model.state_dict(), 
                    "optimizer": optimizer.state_dict(), 
                    "word2id": lang.word2id, 
                    "slot2id": lang.slot2id, 
                    "intent2id": lang.intent2id}

        configurations = f'LR = {lr}\nhid_size = {config.hidden_size}\n\noptimizer={str(type(optimizer))}\nmodel={str(type(model))}\n'
        results_txt = f'{configurations}Intent Accuracy:  {intent_result}\nSlot F1: {f1_result}\nEpochs: {sampled_epochs[-1]}/{n_epochs} ' 
        save_model_incrementally(saving_model,sampled_epochs,losses_train,losses_dev,accuracy_history,results_txt)