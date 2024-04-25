from collections import Counter
import copy
import os
from sklearn.model_selection import train_test_split
from functions import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import PAD_TOKEN, IntentsAndSlots, Lang, collate_fn, load_data
from model import ModelIAS
import numpy as np
import torch.optim as optim



if __name__ == "__main__":


    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough
    device = 'cuda:0'
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

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

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    hid_size = 350
    emb_size = 350

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    accuracy_history = []
    best_f1 = 0
    best_model = None

    runs = 5
    slot_f1s, intent_acc = [], []
    for x in tqdm(range(0, runs)):

        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
        _sampled_epochs = []
        best_f1 = 0
        for x in range(1,n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            criterion_intents, model)
            if x % 5 == 0:
                _sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                            criterion_intents, model, lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']
                accuracy_history.append(intent_res['accuracy'])

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patient
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                                criterion_intents, model, lang)
        intent_acc.append(intent_test['accuracy'])
        slot_f1s.append(results_test['total']['f'])
        sampled_epochs.append(_sampled_epochs)

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))

    configurations = dict()
    configurations['lr'] = lr
    configurations['hid_size'] = hid_size
    configurations['emb_size'] = emb_size
    configurations['optimizer'] = str(type(optimizer))
    configurations['model'] = str(type(model))
    configurations['n_epochs'] = n_epochs
    configurations['patience'] = patience
    configurations['clip'] = clip
    configurations['runs'] = runs

    results = dict()
    results['configurations'] = configurations
    results['slot_f1s_mean'] = round(slot_f1s.mean(),3)
    results['slot_f1s_std'] = round(slot_f1s.std(),3)
    results['intent_acc_mean'] = round(intent_acc.mean(), 3)
    results['intent_acc_std'] = round(slot_f1s.std(), 3)
    results['slot_f1s'] = slot_f1s
    results['intent_acc'] = intent_acc
    results['sampled_epochs'] = sampled_epochs
    results['losses_train'] = losses_train
    results['losses_dev'] = losses_dev
    results['accuracy_history'] = accuracy_history

    save_model_incrementally(best_model,sampled_epochs,losses_train,losses_dev,accuracy_history,results)
