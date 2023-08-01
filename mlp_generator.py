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

class PredictiveEarlyStopping:
    def __init__(self, input_size, hidden_size, num_layers):
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # To output the final accuracy prediction
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.fc.to(self.device)

    def extract_features(self, sequence, initial_accuracies):
        """
        Extracts features from the encoded sequence and initial accuracies.

        :param sequence: Encoded sequence of the architecture.
        :param initial_accuracies: List of validation accuracies for the initial epochs.
        :return: PyTorch tensor representing the features.
        """
        # Combine the sequence and initial accuracies into a single feature vector
        features = sequence + initial_accuracies

        # Convert the features into a PyTorch tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Adding batch dimension

        return features_tensor

        def train(self, features_tensor, target):
            """
            Trains the predictive model using the given features and target.

            :param features_tensor: PyTorch tensor representing the features.
            :param target: Target value (final accuracy).
            :return: Loss value for the training step.
            """
            # Forward pass through the LSTM
            lstm_out, _ = self.model(features_tensor)
            lstm_out = lstm_out[:, -1, :]  # Take the last output of the sequence

            # Pass the LSTM output through the fully connected layer
            prediction = self.fc(lstm_out)

            # Compute the loss
            loss = self.loss_func(prediction, torch.FloatTensor([target]).to(self.device))

            # Perform backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

        def predict(self, features_tensor):
            """
            Predicts the final accuracy using the given features.

            :param features_tensor: PyTorch tensor representing the features.
            :return: Predicted final accuracy.
            """
            # Forward pass through the LSTM
            lstm_out, _ = self.model(features_tensor)
            lstm_out = lstm_out[:, -1, :]  # Take the last output of the sequence

            # Pass the LSTM output through the fully connected layer
            prediction = self.fc(lstm_out)
        
            return prediction.item()



class MLPSearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()
        self.activation_dict = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "relu": nn.ReLU(), "elu": nn.ELU()}
        self.shared_weights = {} # Initialize shared weights attribute
        self.weights_file = 'shared_weights.pkl'

    def set_model_weights(self, model, sequence):
        print("Debugging set_model_weights:")
        print("Sequence:", sequence)
        config_ids = self.get_config_ids(sequence)
        print("Config IDs:", config_ids)

        linear_layer_index = 0
        for i, layer in enumerate(model.children()):
            if isinstance(layer, nn.Linear):
                if linear_layer_index >= len(config_ids):
                    print("Warning: Linear layer index exceeds config_ids length.")
                    break
                config_id = config_ids[linear_layer_index]
                print("Current Linear Layer Index:", linear_layer_index)
                print("Config ID:", config_id)
                if config_id in self.shared_weights:
                    # Print the first three values of the weight matrix before transferring
                    print("Before transferring, first three weight values:", layer.weight.data.view(-1)[:3].tolist())
                    weights, bias = self.shared_weights[config_id]
                    layer.weight.data = torch.tensor(weights).to(self.device)
                    layer.bias.data = torch.tensor(bias).to(self.device)
                    # Print the first three values of the weight matrix after transferring
                    print("After transferring, first three weight values:", layer.weight.data.view(-1)[:3].tolist())
                else:
                    print("Initializing weights for new layer:", config_id)  # New layer
                    # Here you may add the initialization logic if needed

                linear_layer_index += 1
            else:
                print("Skipping non-linear layer")







    def update_weights(self, model, sequence):
        config_ids = self.get_config_ids(sequence)
        linear_layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]

        for i, config_id in enumerate(config_ids):
            print("Updating weights for Config ID", config_id)
            self.shared_weights[config_id] = (linear_layers[i].weight.data.cpu().numpy(), linear_layers[i].bias.data.cpu().numpy())

        with open(self.weights_file, 'wb') as f:
            pickle.dump(self.shared_weights, f)








    def get_config_ids(self, sequence):
        layer_configs = ['input']
        for layer_conf in self.decode_sequence(sequence):
            if layer_conf != 'dropout':
                layer_configs.append((layer_conf[0], layer_conf[1]))
        print("Layer Configs:", layer_configs)  # Print the layer configurations
        config_ids = [(layer_configs[i - 1], layer_configs[i]) for i in range(1, len(layer_configs))]
        return config_ids



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

        self.initial_epochs_data = [] # data for the adaptive stop

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


    def train_model(self, model, data_loader, nb_epochs, sequence, weight_sharing):

        if weight_sharing and sequence:
            self.set_model_weights(model, sequence)
        criterion, optimizer = self.loss_func_and_optimizer(model)

        train_loader = data_loader[0]
        valid_loader = data_loader[1]

        train_losses = []
        train_accs = []
        valid_losses = []
        valid_accs = []

        recent_accuracies = [] # List to keep track of recent validation accuracies

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
            valid_acc = accuracy_score(y_true, y_pred)
            valid_accs.append(valid_acc)

            # Collect data for the first N epochs
            if PREDICTIVE_EARLY_STOPPING and epoch < EARLY_STOPPING_PREDICTIVE_EPOCHS:
                initial_epochs_data.append(valid_acc)
            elif PREDICTIVE_EARLY_STOPPING and epoch == EARLY_STOPPING_PREDICTIVE_EPOCHS:
                # Use the predictive model to make a decision
                prediction_input = initial_epochs_data + self.encode_sequence(sequence)
                predicted_final_acc = predictive_model.predict([prediction_input]) # Note: You need to define predictive_model
                if predicted_final_acc < EARLY_STOPPING_THRESHOLD:
                    print("Early stopping based on predictive model.")
                    break

            t.set_description("Training loss = %f val accuracy = %f" % (train_loss_total / len(train_loader), valid_acc))
            t.refresh() # to show immediately the update

        history = {"accuracy": train_accs, "loss": train_losses, "val_accuracy": valid_accs, "val_loss": valid_losses}
        return history