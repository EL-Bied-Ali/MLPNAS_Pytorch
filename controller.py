import os
# from tabnanny import verbose

from mlp_generator import MLPSearchSpace

from CONSTANTS import *
from rnn_controller import RNNController

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class Controller(MLPSearchSpace):

    def __init__(self):

        self.max_len = MAX_ARCHITECTURE_LENGTH
        self.controller_lstm_dim = CONTROLLER_LSTM_DIM
        self.controller_optimizer = CONTROLLER_OPTIMIZER
        self.controller_lr = CONTROLLER_LEARNING_RATE
        self.controller_decay = CONTROLLER_DECAY
        self.controller_momentum = CONTROLLER_MOMENTUM
        self.use_predictor = CONTROLLER_USE_PREDICTOR

        self.controller_weights = 'LOGS/controller_weights.pth'

        self.seq_data = []

        super().__init__(TARGET_CLASSES)

        self.controller_classes = len(self.vocab) + 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_architecture_sequences(self, model, number_of_samples):
        final_layer_id = len(self.vocab)
        dropout_id = final_layer_id - 1
        samples = []
        log_probs = []
        print("GENERATING ARCHITECTURE SAMPLES...")
        print('------------------------------------------------------')
        while len(samples) < number_of_samples:
            seed = []
            seed_log_probs = []
            while len(seed) < self.max_len:
                sequence = self.pad_sequence_torch([seed], max_len=self.max_len - 1)
                sequence = sequence.reshape(1, self.max_len - 1)
                probab = model(torch.as_tensor(sequence, dtype=int).to(self.device))
                probab = probab[0]
                next_tensor = probab.multinomial(1)
                next = next_tensor.item()  # Convert tensor to integer

                next_log_prob = torch.log(probab[next_tensor])
                if next == dropout_id and len(seed) == 0:
                    continue
                if next == final_layer_id and len(seed) == 0:
                    continue
                if next == final_layer_id:
                    seed.append(next)
                    seed_log_probs.append(next_log_prob)
                    break
                if len(seed) == self.max_len - 1:
                    seed.append(final_layer_id)
                    seed_log_probs.append(next_log_prob)
                    break
                if not next == 0:
                    seed.append(next)
                    seed_log_probs.append(next_log_prob)
            if seed not in self.seq_data:
                samples.append(seed)
                log_probs.append(torch.stack(seed_log_probs).sum())
                self.seq_data.append(seed)
        return samples, log_probs




    def control_model(self):
        
        vocab_size = self.controller_classes
        output_size = self.controller_classes
        embedding_dim = 400
        hidden_dim = self.controller_lstm_dim
        n_layers = 1

        model = RNNController(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
        return model.to(self.device)   

    def train_control_model(self, model, train_loader, loss_func, nb_epochs):
        if self.controller_optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.controller_lr, weight_decay=self.controller_decay, momentum=self.controller_momentum)
        
        else:
            optimizer = getattr(torch.optim, self.controller_optimizer)(model.parameters(), lr=self.controller_lr, weight_decay=self.controller_decay)    

        criterion = loss_func
        # criterion = nn.CrossEntropyLoss()


        if os.path.exists(self.controller_weights):
            model.load_state_dict(torch.load(self.controller_weights))

        
        print("TRAINING CONTROLLER...")

        for i in range(nb_epochs):
    
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                model.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()        
                
        torch.save(model.state_dict(), self.controller_weights)

    # def hybrid_control_model(self, controller_input_shape, controller_batch_size):
    #     if(controller_batch_size == 0):
    #         batch_size = None
    #     else:
    #         batch_size = controller_batch_size
    #     main_input = Input(shape=controller_input_shape, batch_size=batch_size, name='main_input')
    #     x = LSTM(self.controller_lstm_dim, return_sequences=False)(main_input)
    #     predictor_output = Dense(1, activation='sigmoid', name='predictor_output')(x)
    #     main_output = Dense(self.controller_classes, activation='softmax', name='main_output')(x)
    #     model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
    #     return model

    # def train_hybrid_model(self, model, x_data, y_data, pred_target, loss_func, controller_batch_size, nb_epochs):
    #     if self.controller_optimizer == 'sgd':
    #         optim = optimizers.SGD(lr=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
    #     else:
    #         optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
    #     model.compile(optimizer=optim,
    #                   loss={'main_output': loss_func, 'predictor_output': 'mse'},
    #                   loss_weights={'main_output': 1, 'predictor_output': 1})
    #     if os.path.exists(self.controller_weights):
    #         model.load_weights(self.controller_weights)
    #     print("TRAINING CONTROLLER...")
    #     model.fit({'main_input': x_data},
    #               {'main_output': y_data.reshape(len(y_data), 1, self.controller_classes),
    #                'predictor_output': np.array(pred_target).reshape(len(pred_target), 1, 1)},
    #               epochs=nb_epochs,
    #               batch_size=controller_batch_size,
    #               verbose=0)
    #     model.save_weights(self.controller_weights)

    # def get_predicted_accuracies_hybrid_model(self, model, seqs):
    #     pred_accuracies = []
    #     for seq in seqs:
    #         control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
    #         xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
    #         (_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
    #         pred_accuracies.append(pred_accuracy[0])
    #     return pred_accuracies

    def pad_sequence_torch(self, sequences, max_len):
        
        tmp_t = torch.zeros(max_len, dtype=int)
        sequences_t = [torch.as_tensor(l) for l in sequences] + [tmp_t]
        p = pad_sequence(sequences_t) 
        p = p.T
        return p[:-1]
