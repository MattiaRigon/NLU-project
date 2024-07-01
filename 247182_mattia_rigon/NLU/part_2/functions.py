# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import os

from matplotlib import pyplot as plt
import numpy as np
from conll import evaluate
from sklearn.metrics import classification_report
import torch.nn as nn
import torch
import json

from utils import Lang

PAD_TOKEN = 0

def train_loop(data, optimizer, criterion_slots, criterion_intents, model ,clip=5):
    """
    Trains the model using the provided data and optimization parameters.

    Args:
        data (list): A list of samples containing utterances, intents, and slots.
        optimizer: The optimizer used for updating the model's weights.
        criterion_slots: The criterion used for calculating the slot loss.
        criterion_intents: The criterion used for calculating the intent loss.
        model: The model to be trained.
        clip (float, optional): The maximum gradient norm for gradient clipping. Defaults to 5.

    Returns:
        list: A list of loss values for each training sample.
    """
    
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intents = model(sample['utterances'])
        # Compute the loss and softmax
        loss_intent = criterion_intents(intents, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang : Lang, bert_tokenizer):

    """
    Evaluate the performance of a model on a given dataset.

    Args:
        data (list): List of samples to evaluate the model on.
        criterion_slots: Criterion for calculating the loss of slot predictions.
        criterion_intents: Criterion for calculating the loss of intent predictions.
        model: The model to evaluate.
        lang (Lang): Language object containing mappings for intents and slots.
        bert_tokenizer: BERT tokenizer for converting token IDs to tokens.

    Returns:
        tuple: A tuple containing the evaluation results, intent classification report, and loss array.
    """
    
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    with torch.no_grad(): # Avoid computing the gradient
        for sample in data:
            # Forward pass
            slots, intents = model(sample['utterances'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] 
                           for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                # Get the tokens of the utterance
                utterance = bert_tokenizer.convert_ids_to_tokens(sample['utterances']['input_ids'][id_seq])
                length =  len(utterance)
                # Get the ground truth slots ids and convert them to slots
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                # Get the predicted slots ids and convert them to slots
                to_decode = seq[:length].tolist()
                # We need to delete the indexes of the special tokens
                delete_indexes = []
                # We do not want to consider the special tokens
                not_accepted_values = ['pad','[CLS]','[SEP]']
                # Get the indexes of the special tokens
                for i,item in  enumerate(gt_slots):
                    if item in not_accepted_values:
                        delete_indexes.append(i)
                # Remove the special tokens from the ground truth slots
                ref_slots.append([(utterance[id_el], gt_slots[id_el]) for id_el, elem in enumerate(utterance) if gt_slots[id_el] not in not_accepted_values ])

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                real_tmp_slots = []
                # Remove the special tokens from the predicted slots
                for i,slot in enumerate(tmp_seq):
                    if i not in delete_indexes:
                        real_tmp_slots.append(slot)
                # Append the predicted slots
                hyp_slots.append(real_tmp_slots)
    try:            
        # Evaluate the model for slot filling
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
    # Evaluate the model for intent classification
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_weights(mat):
    """
    Initializes the weights of the given module using Xavier initialization for linear layers
    and orthogonal initialization for recurrent layers.

    Args:
        mat (nn.Module): The module for which to initialize the weights.

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
    torch.save(model, model_path)
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
