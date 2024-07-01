# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import os

from matplotlib import pyplot as plt
import numpy as np
from evals import evaluate_ote
from sklearn.metrics import classification_report
import torch.nn as nn
import torch
import json

from utils import Lang

PAD_TOKEN = 0

def train_loop(data, optimizer, criterion_slots, model ,clip=5):
    """
    Trains the model using the given data, optimizer, criterion, and model.

    Args:
        data (list): List of samples containing 'sentences' and 'y_slots' keys.
        optimizer: The optimizer used to update the model's weights.
        criterion_slots: The criterion used to compute the loss between predicted slots and ground truth slots.
        model: The model to be trained.
        clip (float, optional): The maximum gradient norm to clip. Defaults to 5.

    Returns:
        list: List of loss values for each sample in the data.
    """

    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots = model(sample['sentences'])
        # Compute the loss and softmax
        loss = criterion_slots(slots, sample['y_slots'])
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def get_weights(dataset):
    '''
    Get the weights for the label in the dataset
    '''
    count = 0
    length = 0
    for sample in dataset:
        for slot in sample['slots']:
            length += 1
            if 2 == int(slot):
                count += 1
    return [1, count/length, (length-count)/length]

def eval_loop(data, criterion_slots, model, bert_tokenizer):
    """
    Evaluate the model on the given data.

    Args:
        data (list): List of samples to evaluate the model on.
        criterion_slots (torch.nn.Module): Loss criterion for slot prediction.
        model (torch.nn.Module): Model to evaluate.
        bert_tokenizer (transformers.PreTrainedTokenizer): Tokenizer for BERT model.

    Returns:
        tuple: A tuple containing the evaluation results and the array of loss values.

    """

    model.eval()
    loss_array = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots = model(sample['sentences'])
            
            loss = criterion_slots(slots, sample['y_slots'])
            loss_array.append(loss.item())
    
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                # Get the sentence from the input_ids
                sentence = bert_tokenizer.convert_ids_to_tokens(sample['sentences']['input_ids'][id_seq])
                length =  len(sentence)
                # Get the ground truth slots
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [elem for elem in gt_ids[:length]]
                # get the predicted slots
                to_decode = seq[:length].tolist()
                # Remove the padding
                delete_indexes = []
                for i,item in  enumerate(gt_slots):
                    if item == 0:
                        delete_indexes.append(i)
                # remove the padding from the ground truth
                ref_slots.append([gt_slots[id_el] for id_el, elem in enumerate(sentence) if gt_slots[id_el] != 0])

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append(elem)
                real_tmp_slots = []
                # remove the prediction in the deleted indexes
                for i,slot in enumerate(tmp_seq):
                    if i not in delete_indexes:
                        real_tmp_slots.append(slot)
                hyp_slots.append(real_tmp_slots)
    try:        
        # Evaluate the model    
        results = evaluate_ote(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    return results, loss_array

def init_weights(mat):
    """
    Initializes the weights of the given module using specific initialization methods.

    Args:
        mat (nn.Module): The module for which the weights need to be initialized.

    Returns:
        None
    """

    for n, m in mat.named_modules():
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
                if 'slot_out' in n or 'intent_out' in n:
                    torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias != None:
                        m.bias.data.fill_(0.01)

def save_model_incrementally(model, sampled_epochs, losses_train, losses_dev, accuracy_history, results, model_name='model.pth'):
    """
    Save the model and related information incrementally.

    Args:
        model (torch.nn.Module): The model to be saved.
        sampled_epochs (list): List of sampled epochs.
        losses_train (list): List of training losses.
        losses_dev (list): List of development losses.
        accuracy_history (list): List of accuracy history.
        results (str): Results to be saved.
        model_name (str, optional): Name of the model file. Defaults to 'model.pth'.
    """

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
    save_plot_accuracy(sampled_epochs,accuracy_history,new_test_dir)
    with open(os.path.join(new_test_dir,"result.txt"), 'w') as file:
        file.write(results)
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

    plt.figure(figsize=(10, 6)) 

    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o') 
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='s')  

    plt.title('Training and Validation Loss')  
    plt.xlabel('Epochs') 
    plt.ylabel('Loss')  
    plt.legend()

    plt.grid(True)  
    plt.tight_layout()  

    plt.savefig(os.path.join(path, "losses.png"))

def save_plot_accuracy(sampled_epochs, accuracy_history, path):
    """
    Save a plot of accuracy history.

    Args:
        sampled_epochs (list): List of sampled epochs.
        accuracy_history (list): List of accuracy values.
        path (str): Path to save the plot.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    plt.plot(sampled_epochs, accuracy_history, label='accuracy', marker='o')
    plt.title('PPL')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(path, "accuracy.png"))
