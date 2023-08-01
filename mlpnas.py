import pickle
from CONSTANTS import *
from controller import Controller
from mlp_generator import MLPGenerator
from utils import *

import pickle
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset


class MLPNAS(Controller):

    def __init__(self, data_loader):
        self.data_loader = load_dataset()
        # self.data_loader = data_loader
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA
        self.weight_sharing = WEIGHT_SHARING
        self.weight_sharing_threshold = WEIGHT_SHARING_THRESHOLD
            
        self.weight_sharing = WEIGHT_SHARING
        self.weights_file = 'shared_weights.pkl' # Path to save shared weights
        if self.weight_sharing:
            if os.path.exists(self.weights_file):
                with open(self.weights_file, 'rb') as f:
                    self.shared_weights = pickle.load(f)
            else:
                print("Initializing shared weights dictionary...")
                self.shared_weights = {}




        x = next(iter(data_loader[0]))
        x = torch.as_tensor(x[0])
        self.input_shape = x.shape #included batch_size

        self.data = []
        # Create a new directory for each dataset and method, if it doesn't already exist
        log_dir = os.path.join('LOGS', DATASET_CHOICE, METHOD)
        os.makedirs(log_dir, exist_ok=True)

        # Find the next available ID for the log file
        nas_data_id = 1
        while os.path.exists(os.path.join(log_dir, f'nas_data_{nas_data_id}{"_ws" if self.weight_sharing else ""}.pkl')):
            nas_data_id += 1

        # Set the log file path
        self.nas_data_log = os.path.join(log_dir, f'nas_data_{nas_data_id}{"_ws" if self.weight_sharing else ""}.pkl')


        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (MAX_ARCHITECTURE_LENGTH - 1, 1)
        self.controller_model = self.control_model()

        # Create the optimizer object
        if self.controller_optimizer == 'sgd':
            self.controller_optimizer = torch.optim.SGD(self.controller_model.parameters(), lr=self.controller_lr, weight_decay=self.controller_decay, momentum=self.controller_momentum)
        else:
            self.controller_optimizer = getattr(torch.optim, self.controller_optimizer)(self.controller_model.parameters(), lr=self.controller_lr, weight_decay=self.controller_decay)


    
            
    def create_architecture(self, sequence):
        model = self.model_generator.create_model(sequence, self.input_shape)
    
        if self.weight_sharing: # Check if weight sharing is enabled
            self.model_generator.set_model_weights(model, sequence) # Transfer shared weights

        return model





    def train_architecture(self, model, sequence):
        history = self.model_generator.train_model(model, self.data_loader, self.architecture_train_epochs, sequence, self.weight_sharing)
    
        if self.weight_sharing and history['val_accuracy'][-1] >= self.weight_sharing_threshold:
            print("Saving shared weights...") # Debug print
            self.model_generator.update_weights(model, sequence) # Update shared weights

        return history







    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        val_acc = 0
        if len(history['val_accuracy']) == 1:
            val_acc = history['val_accuracy'][0]
        else:
            val_acc = np.ma.average(history['val_accuracy'],
                                    weights=np.arange(1, len(history['val_accuracy']) + 1),
                                    axis=-1)
        # Get the loss for the architecture
        loss = history['loss'][-1]  # The loss after the last training epoch
    
        # Append the loss to the data
        if pred_accuracy:
            self.data.append([sequence, val_acc, pred_accuracy, loss])
        else:
            self.data.append([sequence, val_acc, loss])
    
        # Save the data to the log file
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)


    def pad_sequence_torch(self, sequences, max_len):
        
        tmp_t = torch.zeros(max_len, dtype=int)
        sequences_t = [torch.as_tensor(l) for l in sequences] + [tmp_t]
        p = pad_sequence(sequences_t) 
        p = p.T
        return p[:-1]
    
    
    def prepare_controller_data(self, sequences):

        controller_sequences = self.pad_sequence_torch(sequences, max_len=self.max_len)

        xc = torch.as_tensor(controller_sequences[:, :-1].reshape(len(controller_sequences), 2), dtype=int)
        yc = F.one_hot(torch.as_tensor(controller_sequences[:, -1], dtype=int), self.controller_classes)

        train_data = TensorDataset(xc, yc)
        train_loader = DataLoader(train_data, shuffle=False, batch_size=len(sequences))
        
        val_acc_target = [item[1] for item in self.data]
        return train_loader, val_acc_target

    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, output, target):
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        discounted_reward = torch.as_tensor(discounted_reward).to(self.device)
        loss = -torch.matmul(torch.log(output.T.to(self.device)) , discounted_reward)
        loss = loss.sum()
        return loss

    def train_controller(self, model, train_loader, pred_accuracy=None):
        if self.use_predictor:
            pass
            # self.train_hybrid_model(model,
            #                         x,
            #                         y,
            #                         pred_accuracy,
            #                         self.custom_loss,
            #                         len(self.data),
            #                         self.controller_train_epochs) 
        else:
            self.train_control_model(model,
                                     train_loader,
                                     self.custom_loss,
                                     self.controller_train_epochs)

    def search(self, test=True):
        vocab = self.vocab_dict()
        valid_ids = list(vocab.keys())
        dropout_id = valid_ids[-2]
        final_layer_id = valid_ids[-1]

        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')

            # For testing weight sharing, generate a specific architecture in every second epoch
            if test:
                if controller_epoch == 0:
                    # First architecture for testing
                    sequence = [14, 2, 30]  # Example architecture 1
                elif controller_epoch == 1:
                    # Second architecture for testing (slightly different from the first one)
                    sequence = [12, 2, 30]  # Example architecture 2
                sequences = [sequence]
                log_probs = [0]
            elif METHOD == 'random_search':
                # Generate a random architecture sequence that follows the rules
                sequence = []
                while len(sequence) < self.max_len:
                    next = np.random.choice(valid_ids[:-1])  # Exclude final_layer_id from random choice
                    if next == dropout_id and len(sequence) == 0:
                        continue  # Skip if first layer is dropout
                    sequence.append(next)
                    if len(sequence) == self.max_len - 1:
                        sequence.append(final_layer_id)  # Add final layer as the last layer
                        break
                sequences = [sequence]
                log_probs = [0]  # log_probs is not used in the random search case, but we'll set it to avoid errors later
            else:
                # Sample architecture sequences using the controller model
                sequences, log_probs = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)

            rewards = []
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model, sequence)

                self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
                # Get the reward for the action
                reward = -history['val_accuracy'][-1] # Assuming that the validation accuracy is stored in the 'val_accuracy' key of the history dictionary
                rewards.append(reward)
                if self.weight_sharing and history['val_accuracy'][-1] < WEIGHT_SHARING_THRESHOLD:
                    continue

            # Calculate the policy loss
            policy_loss = []
            if METHOD == 'vanilla':
                for log_prob, reward in zip(log_probs, rewards):
                    policy_loss.append(-log_prob * reward)
            elif METHOD == 'adaptive_baseline':
                baseline = np.mean(rewards)
                for log_prob, reward in zip(log_probs, rewards):
                    policy_loss.append(-log_prob * (reward - baseline))
            if METHOD != 'random_search':
                policy_loss = torch.stack(policy_loss).sum()
                # Perform a gradient update
                self.controller_optimizer.zero_grad()
                policy_loss.backward()
                self.controller_optimizer.step()

        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
