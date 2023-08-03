########################################################
#                   NAS PARAMETERS                     #
########################################################
"""
CONTROLLER_SAMPLING_EPOCHS = 2
SAMPLES_PER_CONTROLLER_EPOCH = 2
CONTROLLER_TRAINING_EPOCHS = 2
ARCHITECTURE_TRAINING_EPOCHS = 1
CONTROLLER_LOSS_ALPHA = 0.9

"""
CONTROLLER_SAMPLING_EPOCHS = 10
SAMPLES_PER_CONTROLLER_EPOCH = 20
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 10
CONTROLLER_LOSS_ALPHA = 0.9


METHOD = 'random_search' # vanilla adaptive_search random_search
WEIGHT_SHARING = False  # Set this to True to enable weight sharing
WEIGHT_SHARING_THRESHOLD = 0.5  # Set this to the minimum accuracy for an architecture to have its weights shared

########################################################
#               PREDICTIVE EARLY STOPPING              #
########################################################
EARLY_STOPPING_PREDICTIVE_EPOCHS = 5
PREDICTIVE_EARLY_STOPPING = False
if PREDICTIVE_EARLY_STOPPING:
    SAMPLES_PER_CONTROLLER_EPOCH = 20
# LSTM parameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1




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
DATASET_CHOICE = 'MNIST'  # or 'MNIST' or 'Fashion-MNIST' CIFAR-10
TARGET_CLASSES = 10
VALIDATION_SPLIT_RATE = 0.1
MANUAL_SEED = 1
BATCH_SIZE = 512
NUM_WORKERS = 1

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 5
