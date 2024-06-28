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

def post_process_model(intent_logits,slot_logits,intent_label_ids, slot_labels_ids,attention_mask,num_intent_labels,num_slot_labels,ignore_index,outputs):
    slot_loss_coef = 1
    total_loss = 0
    # 1. Intent Softmax
    if intent_label_ids is not None:
        if num_intent_labels == 1:
            intent_loss_fct = nn.MSELoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
        else:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, num_intent_labels), intent_label_ids.view(-1))
        total_loss += intent_loss

    # 2. Slot Softmax
    if slot_labels_ids is not None:

        slot_loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_logits.view(-1, num_slot_labels)[active_loss]
            active_labels = slot_labels_ids.view(-1)[active_loss]
            slot_loss = slot_loss_fct(active_logits, active_labels)
        else:
            slot_loss = slot_loss_fct(slot_logits.view(-1, num_slot_labels), slot_labels_ids.view(-1))
        total_loss += slot_loss_coef * slot_loss

    outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

    # outputs = (total_loss,) + outputs

    return total_loss, outputs



def train_loop(data, optimizer, criterion_slots, criterion_intents, model ,clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intents = model(sample['utterances'])
        # Compute the loss and softmax
        loss_intent = criterion_intents(intents, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang : Lang, bert_tokenizer):
    model.eval()
    loss_array = []
    
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
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
                # length = sample['slots_len'].tolist()[id_seq]
                # utt_ids = sample['utterance'][id_seq][:length].split(" ")

                utterance = bert_tokenizer.convert_ids_to_tokens(sample['utterances']['input_ids'][id_seq])
                # utterance = [u for u in utterance if u not in ['[CLS]','[SEP]','[PAD]']]
                length =  len(utterance)# torch.sum(sample['utterances']['attention_mask'][0] == 1).item() - 2 
                
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                            
                to_decode = seq[:length].tolist()
                delete_indexes = []
                not_accepted_values = ['pad','[CLS]','[SEP]']

                for i,item in  enumerate(gt_slots):
                    if item in not_accepted_values:
                        delete_indexes.append(i)


                ref_slots.append([(utterance[id_el], gt_slots[id_el]) for id_el, elem in enumerate(utterance) if gt_slots[id_el] not in not_accepted_values ])

                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                real_tmp_slots = []

                for i,slot in enumerate(tmp_seq):
                    if i not in delete_indexes:
                        real_tmp_slots.append(slot)

                hyp_slots.append(real_tmp_slots)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, 
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_weights(mat):
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

def save_model_incrementally(model, sampled_epochs, losses_train, losses_dev, accuracy_history, results,model_name='model.pth'):
    """
    Saves a PyTorch model in an incrementally named test folder within a results directory.

    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - model_name (str, optional): The name of the saved model file. Default is 'model.pth'.
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
    

def save_plot_losses(sampled_epochs,losses_train,losses_dev,path):

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

def save_plot_accuracy(sampled_epochs,accuracy_history,path):

    plt.figure(figsize=(10, 6)) 
    plt.plot(sampled_epochs, accuracy_history, label='accuracy', marker='o')  
    plt.title('PPL')  
    plt.xlabel('Epochs')  
    plt.ylabel('Loss')  
    plt.legend() 

    plt.grid(True) 
    plt.tight_layout()  

    plt.savefig(os.path.join(path,"accuracy.png")) 