import pickle
from CONSTANTS import *
from controller import Controller
from mlp_generator import MLPGenerator
from utils import *

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

        x = next(iter(data_loader[0]))
        x = torch.as_tensor(x[0])
        self.input_shape = x.shape #included batch_size

        self.data = []
        # Create a new directory for each dataset and method, if it doesn't already exist
        log_dir = os.path.join('LOGS', DATASET_CHOICE, METHOD)
        os.makedirs(log_dir, exist_ok=True)

        # Find the next available ID for the log file
        nas_data_id = 1
        while os.path.exists(os.path.join(log_dir, f'nas_data_{nas_data_id}.pkl')):
            nas_data_id += 1

        # Set the log file path
        self.nas_data_log = os.path.join(log_dir, f'nas_data_{nas_data_id}.pkl')

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
        return model

    def train_architecture(self, model):
        history = self.model_generator.train_model(model, self.data_loader, self.architecture_train_epochs)
        return history

    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        if len(history['val_accuracy']) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                  history['val_accuracy'][0],
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  history['val_accuracy'][0]])
            print('validation accuracy: ', history['val_accuracy'][0])
        else:
            val_acc = np.ma.average(history['val_accuracy'],
                                    weights=np.arange(1, len(history['val_accuracy']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                  val_acc,
                                  pred_accuracy])
            else:
                self.data.append([sequence,
                                  val_acc])
            print('validation accuracy: ', val_acc)

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

    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences, log_probs = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            rewards = []
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
                # Get the reward for the action
                reward = -history['val_accuracy'][-1] # Assuming that the validation accuracy is stored in the 'val_accuracy' key of the history dictionary
                rewards.append(reward)
            # Calculate the policy loss
            policy_loss = []
            if METHOD == 'vanilla':
                for log_prob, reward in zip(log_probs, rewards):
                    policy_loss.append(-log_prob * reward)
            elif METHOD == 'adaptive_baseline':
                baseline = np.mean(rewards)
                for log_prob, reward in zip(log_probs, rewards):
                    policy_loss.append(-log_prob * (reward - baseline))
            policy_loss = torch.stack(policy_loss).sum()
            # Perform a gradient update
            self.controller_optimizer.zero_grad()
            policy_loss.backward()
            self.controller_optimizer.step()
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data



