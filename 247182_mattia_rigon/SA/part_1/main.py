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
import argparse

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test application")
    parser.add_argument('--test',type=str, help='If test enabled just evaluate the model with model.pth, otherwise train')
    args = parser.parse_args()
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

    words = sum([x['sentence'] for x in train_raw], []) # No set() since we want to compute 
                                                            # the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                            # however this depends on the research purpose
    slots = set(sum([line['slots'] for line in corpus],[]))

    train_dataset = Slots(train_raw)
    dev_dataset = Slots(dev_raw)
    test_dataset = Slots(test_raw,)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=lambda x: collate_fn(x), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=lambda x: collate_fn(x))

    lr = 0.00005 # learning rate
    clip = 5 # Clip the gradient
    out_slot = 3
    config = BertConfig.from_pretrained('bert-base-uncased')
    model = SABert(config=config,out_slot=out_slot, dropout=0.1)
    model.to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    n_epochs = 7
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    accuracy_history = []
    best_f1 = -1
    best_model = None

    if args.test:
        try:
            saved_model = torch.load(os.path.join(LOCAL_PATH,'bin',args.test))
        except Exception as e:
            print(f"Error occured reading the weights: {e}.")
            sys.exit(1)
        model.load_state_dict(saved_model)
        results_test, _ = eval_loop(test_loader, criterion_slots, model,tokenizer)
        print('Aspect Precision', results_test[0])
        print('Aspect Recall', results_test[1])
        print('Aspect F1', results_test[2])
    else:
        for x in tqdm(range(1,n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, 
                            model, clip=clip)
            if x % 1 == 0: 
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, loss_dev = eval_loop(dev_loader, criterion_slots,model,tokenizer)
                losses_dev.append(np.asarray(loss_dev).mean())
                
                f1 = results_dev[2]
                accuracy_history.append(f1)
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
        results_test, _ = eval_loop(test_loader, criterion_slots,best_model,tokenizer)   
        print('Aspect Precision', results_test[0])
        print('Aspect Recall', results_test[1])
        print('Aspect F1', results_test[2])
        configurations = f'LR = {lr}\nhid_size = {config.hidden_size}\n\noptimizer={str(type(optimizer))}\nmodel={str(type(model))}\n'
        results_txt = f'{configurations}\nSlot F1: {results_test[2]}\nEpochs: {sampled_epochs[-1]}/{n_epochs} ' 
        save_model_incrementally(best_model,sampled_epochs,losses_train,losses_dev,accuracy_history,results_txt)