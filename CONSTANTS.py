########################################################
#                   NAS PARAMETERS                     #
########################################################

CONTROLLER_SAMPLING_EPOCHS = 2
SAMPLES_PER_CONTROLLER_EPOCH = 3
CONTROLLER_TRAINING_EPOCHS = 2
ARCHITECTURE_TRAINING_EPOCHS = 1
CONTROLLER_LOSS_ALPHA = 0.9

"""
CONTROLLER_SAMPLING_EPOCHS = 10
SAMPLES_PER_CONTROLLER_EPOCH = 10
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 10
CONTROLLER_LOSS_ALPHA = 0.9
"""

METHOD = 'random_search' # vanilla constant_baseline random_search
WEIGHT_SHARING = True  # Set this to True to enable weight sharing
WEIGHT_SHARING_THRESHOLD = 0.01  # Set this to the minimum accuracy for an architecture to have its weights shared


########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = False

########################################################
#                   MLP PARAMETERS                     #
########################################################
MAX_ARCHITECTURE_LENGTH = 3
MLP_OPTIMIZER = 'Adam'
MLP_LEARNING_RATE = 0.01
MLP_DECAY = 0.0
MLP_MOMENTUM = 0.0
MLP_DROPOUT = 0.2
MLP_LOSS_FUNCTION = 'categorical_crossentropy'

########################################################
#                   DATA PARAMETERS                    #
########################################################
DATASET_CHOICE = 'CIFAR-10'  # or 'MNIST' or 'Fashion-MNIST'
TARGET_CLASSES = 10
VALIDATION_SPLIT_RATE = 0.1
MANUAL_SEED = 1
BATCH_SIZE = 512
NUM_WORKERS = 1

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 5
