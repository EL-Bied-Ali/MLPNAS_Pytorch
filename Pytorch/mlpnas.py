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

        self.data_loader = data_loader
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
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (MAX_ARCHITECTURE_LENGTH - 1, 1)
        # if self.use_predictor:
        #     self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)
        # else:
        self.controller_model = self.control_model()

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
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
            
            train_loader, val_acc_target = self.prepare_controller_data(sequences)
            
            self.train_controller(self.controller_model,
                                  train_loader,
                                  val_acc_target[-self.samples_per_controller_epoch:])
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        return self.data
