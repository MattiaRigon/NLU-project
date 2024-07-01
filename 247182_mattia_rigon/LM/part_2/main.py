# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from torch.utils.data import DataLoader
from utils import Lang, PennTreeBank, collate_fn, read_file
from functools import partial
from model import LSTM
import numpy as np
import argparse
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test application")
    parser.add_argument('--test',type=str, help='If test enabled just evaluate the model with model.pth, otherwise train')
    args = parser.parse_args()

    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough

    batch_train = 20
    batch_dev_test = 80

    train_raw = read_file(os.path.join(LOCAL_PATH,"dataset/PennTreeBank/ptb.train.txt"))
    dev_raw = read_file(os.path.join(LOCAL_PATH,"dataset/PennTreeBank/ptb.valid.txt"))
    test_raw = read_file(os.path.join(LOCAL_PATH,"dataset/PennTreeBank/ptb.test.txt"))

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # Experiment also with a smaller or bigger model by changing hid and emb sizes
    # A large model tends to overfit
    hid_size = 650
    emb_size = 650
    emb_dropout = 0.6
    out_dropout = 0.6
    nonMono = 5
    wdecay = 1.2e-6

    lr = 10 
    clip = 5 # Clip the gradient
    device = 'cuda:0'

    vocab_len = len(lang.word2id)
    average_seq_len =  calculate_average_seq_len(train_loader)

    model = LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"],emb_dropout=emb_dropout,out_dropout=emb_dropout).to(device)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay= wdecay)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    ppl_history = []
    best_ppl = math.inf
    best_model = None
    if args.test:
        try:
            saved_model = torch.load(os.path.join(LOCAL_PATH,'bin',args.test))
        except Exception as e:
            print(f"Error occured reading the weights: {e}.")
            sys.exit(1)
        model.load_state_dict(saved_model)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl: ', final_ppl)
    else:
        pbar = tqdm(range(1,n_epochs))
        #If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, average_seq_len, clip=clip)
            if epoch % 1 == 0:
                if epoch % 5 == 0 :
                    if (lr *0.75) < 1.5 :
                        lr = 1.5
                    else:
                        lr = lr * 0.75
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                if isinstance(optimizer,torch.optim.ASGD):
                    tmp = {}
                    for prm in model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = optimizer.state[prm]['ax'].clone()
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                ppl_history.append(ppl_dev)
                if  ppl_dev < best_ppl: # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 5
                else:
                    patience -= 1

                if isinstance(optimizer,torch.optim.ASGD):
                    for prm in model.parameters():
                        prm.data = tmp[prm].clone()

                if patience <= 0: # Early stopping with patience
                    if isinstance(optimizer,torch.optim.SGD):
                        print('Switching to ASGD')
                        optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=wdecay)
                        patience = 5
                    else:
                        break # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl,  _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)

        configurations = f'LR = {lr}\nhid_size = {hid_size}\nemb_size={emb_size}\noptimizer={str(type(optimizer))}\nmodel={str(type(model))}\nemb_dropout={emb_dropout}\nout_dropout={out_dropout}\nwdecay={wdecay}\nnonMono={nonMono}\nbatch{batch_train}\n'
        results_txt = f'{configurations}Test ppl:  {final_ppl} + Epochs: {sampled_epochs[-1]}/{n_epochs} ' 

        save_model_incrementally(best_model,sampled_epochs,losses_train,losses_dev,ppl_history,results_txt)

