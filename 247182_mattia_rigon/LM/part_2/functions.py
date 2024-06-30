import math
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import copy

def train_loop(data, optimizer, criterion, model, average_seq_len,clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    original_lr = copy.deepcopy(optimizer.param_groups[0]['lr'])

    for sample in data:
        seq_len = sample['source'].shape[1]
        seq_len_adjusted_lr = original_lr * (seq_len / average_seq_len)  # average_seq_len deve essere definito o calcolato
        optimizer.param_groups[0]['lr'] = seq_len_adjusted_lr  # Impostare il learning rate modificato
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) # clip the gradient to avoid explosioning gradients
        optimizer.step() # Update the weights
        optimizer.param_groups[0]['lr'] = original_lr  # Ripristinare il learning rate originale

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def save_model_incrementally(model, sampled_epochs, losses_train, losses_dev, ppl_history, results_txt,model_name='model.pth'):
    """
    Saves a PyTorch model in an incrementally named test folder within a results directory.

    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - sampled_epochs (list): List of sampled epochs during training.
    - losses_train (list): List of training losses corresponding to each sampled epoch.
    - losses_dev (list): List of validation losses corresponding to each sampled epoch.
    - ppl_history (list): List of perplexity values corresponding to each sampled epoch.
    - results_txt (str): Text to be saved in the result.txt file.
    - model_name (str, optional): The name of the saved model file. Default is 'model.pth'.
    """
    # Ensure the 'results' directory exists
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    # Find the next available folder name
    test_folder_name = 'test0'
    i = 0
    while os.path.exists(os.path.join(results_dir, test_folder_name)):
        i += 1
        test_folder_name = f'test{i}'
    # Create the new test directory
    new_test_dir = os.path.join(results_dir, test_folder_name)
    os.makedirs(new_test_dir)
    
    save_plot_losses(sampled_epochs,losses_train,losses_dev,new_test_dir)
    save_plot_ppl(sampled_epochs,ppl_history,new_test_dir)
    with open(os.path.join(new_test_dir,"result.txt"), 'w') as file:
        file.write(results_txt)
    # Save the model in the new directory
    model_path = os.path.join(new_test_dir, model_name)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def save_plot_losses(sampled_epochs, losses_train, losses_dev, path):
    """
    Save a plot of training and validation losses.

    Args:
        sampled_epochs (list): List of sampled epochs.
        losses_train (list): List of training losses.
        losses_dev (list): List of validation losses.
        path (str): Path to save the plot.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import os

    plt.figure(figsize=(10, 6))

    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='s')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(path,"losses.png"))

def save_plot_ppl(sampled_epochs, ppl_history, path):
    """
    Save a plot of perplexity (PPL) over epochs.

    Args:
        sampled_epochs (list): List of sampled epochs.
        ppl_history (list): List of perplexity values.
        path (str): Path to save the plot.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6)) 
    plt.plot(sampled_epochs, ppl_history, label='PPL', marker='o')  
    plt.title('PPL')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend() 

    plt.grid(True) 
    plt.tight_layout()  

    plt.savefig(os.path.join(path,"ppl.png")) 


def calculate_average_seq_len(data_loader):
    total_length = 0
    count = 0
    
    for batch in data_loader:
        # Assumiamo che `batch` sia un dizionario con le chiavi 'source' e 'target'
        # dove 'source' è il tensore contenente le sequenze di input
        batch_seq_lengths = batch['source'].shape[1]  # Prendi la dimensione della sequenza, assumendo che le sequenze siano allineate sulla seconda dimensione
        
        total_length += batch_seq_lengths * batch['source'].shape[0]  # Moltiplica la lunghezza della sequenza per il numero di elementi nel batch
        count += batch['source'].shape[0]  # Conta il numero totale di sequenze
    
    # Calcola la lunghezza media delle sequenze
    average_seq_len = total_length / count if count != 0 else 0
    return average_seq_len