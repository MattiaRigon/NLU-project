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
import torch.optim as optim



if __name__ == "__main__":


    # Dataloader instantiation
    # You can reduce the batch_size if the GPU memory is not enough

    batch_train = 20
    batch_dev_test = 80

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=batch_train, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_dev_test, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    hid_size = 200
    emb_size = 300

    lr = 0.0001 # learning rate
    clip = 5 # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token
    n_epochs = 200
    patience = 3
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(train_loader, optimizer, criterion_slots, 
                        criterion_intents, model, clip=clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, 
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, 
                                            criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.savefig('results/train_dev_losses.png')
    plt.show()
