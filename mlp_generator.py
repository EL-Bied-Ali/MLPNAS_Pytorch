import os
import warnings
from matplotlib.rcsetup import validate_float
import pandas as pd

import torch
import torch.nn as nn
from functools import reduce

from sklearn.metrics import accuracy_score

from CONSTANTS import *
from utils import *
from tqdm import tqdm, trange

class MLPSearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()
        self.activation_dict = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU(), "elu": nn.ELU()}

    def vocab_dict(self):
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        vocab[len(vocab) + 1] = (self.target_classes, 'last layer')

        return vocab

    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])
        return encoded_sequence

    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence


class MLPGenerator(MLPSearchSpace):

    def __init__(self):

        self.target_classes = TARGET_CLASSES
        self.mlp_optimizer = MLP_OPTIMIZER
        self.mlp_lr = MLP_LEARNING_RATE
        self.mlp_decay = MLP_DECAY
        self.mlp_momentum = MLP_MOMENTUM
        self.mlp_dropout = MLP_DROPOUT
        # self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.metrics = ['accuracy']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Target device: " + str(device))

        super().__init__(TARGET_CLASSES)


    #creating a Pytorch model
    def create_model(self, sequence, mlp_input_shape):

        layer_configs = self.decode_sequence(sequence)
        flattened_input_size = [mlp_input_shape[0], reduce((lambda x, y: x * y), mlp_input_shape[1:])] #first element is batch_size so flatten the rest

        model = nn.Sequential()

        if len(mlp_input_shape) > 1:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0 and layer_conf == 'dropout':
                    continue  # Skip if first layer is dropout
                if layer_conf == 'dropout':
                    model.add_module('dropout' + str(i), nn.Dropout(p=self.mlp_dropout))
                else:
                    if(i == 0):
                        model.add_module('linear' + str(i), nn.Linear(flattened_input_size[1], layer_conf[0]))
                        model.add_module('activation' + str(i), self.activation_dict[layer_conf[1]])
                    else:
                        if(layer_configs[i-1] == 'dropout'):
                            prev_layer_conf = layer_configs[i-2]
                        else:
                            prev_layer_conf = layer_configs[i-1]
                        model.add_module('linear' + str(i), nn.Linear(prev_layer_conf[0], layer_conf[0]))
                        if(i < len(layer_configs) - 1):
                            model.add_module('activation' + str(i), self.activation_dict[layer_conf[1]])

            # Ensure last layer is a softmax with 10 nodes
            model.add_module('linear_last', nn.Linear(layer_configs[-1][0], self.target_classes))
            model.add_module('softmax', nn.Softmax(dim=1))
        else:
            assert "Input Size Error!"

        return model.to(self.device)









    def loss_func_and_optimizer(self, model):
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        if self.mlp_optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_decay, momentum=self.mlp_momentum)
        
        else:
            optimizer = getattr(torch.optim, self.mlp_optimizer)(model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_decay)
        
        return criterion, optimizer


    def train_model(self, model, data_loader, nb_epochs):
        
        criterion, optimizer = self.loss_func_and_optimizer(model)

        train_loader = data_loader[0]
        valid_loader = data_loader[1]

        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []

        t = trange(nb_epochs, desc="Training loss", leave=True)

        for epoch in t:

            train_loss_total = 0.0
    
            # ensure model is in training mode
            model.train()    
            
            for data in train_loader:        
                inputs, target = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                optimizer.zero_grad()   
                        
                # forward pass
                output = model(inputs.view(inputs.shape[0], -1)) # flattening the input image
                loss = criterion(output, target)
                
                # backward pass + run optimizer to update weights
                loss.backward()
                optimizer.step()
            
                # keep track of loss value
                train_loss_total += loss.item()

            train_losses.append(train_loss_total / len(train_loader)) #average train loss for each epoch  
            
            # Validation loop
            # valid_loss_total = 0.0
            model.eval()
            y_true = []
            y_pred = []
            for data in valid_loader:
                inputs, target = data
                inputs, target = inputs.to(self.device), target.to(self.device)

                # forward pass
                output = model(inputs.view(inputs.shape[0], -1)) # flattening the input image
                
                # loss = criterion(output, target)
                # valid_loss_total += loss.item()
                
                _, pred = torch.max(output, 1)
                target = target.cpu().float()
                y_true.extend(target.tolist()) 
                y_pred.extend(pred.reshape(-1).tolist())
            
            # valid_losses.append(valid_loss_total / len(valid_loader))
            valid_accs.append(accuracy_score(y_true, y_pred))

            
            
            t.set_description("Training loss = %f val accuracy = %f" % (train_loss_total / len(train_loader), accuracy_score(y_true, y_pred)))
            t.refresh() # to show immediately the update  

        history = {"accuracy": train_accs, "loss": train_losses, "val_accuracy": valid_accs, "val_loss": valid_losses}
        return history