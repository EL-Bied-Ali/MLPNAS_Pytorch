import os
import shutil
import pickle
import numpy as np
from itertools import groupby
from matplotlib import pyplot as plt

from CONSTANTS import *
from mlp_generator import MLPSearchSpace


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch import Generator

########################################################
#                   DATA PROCESSING                    #
########################################################


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


########################################################
#                       LOGGING                        #
########################################################


def clean_log():
    filelist = os.listdir('LOGS') 
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))


def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)


def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))


########################################################
#                 RESULTS PROCESSING                   #
########################################################


def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def sort_search_data(nas_data):
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

########################################################
#                EVALUATION AND PLOTS                  #
########################################################

def get_top_n_architectures(n):
    data = load_nas_data()
    data = sort_search_data(data)
    search_space = MLPSearchSpace(TARGET_CLASSES)
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])


def get_nas_accuracy_plot():
    data = load_nas_data()
    accuracies = [x[1] for x in data]
    plt.plot(np.arange(len(data)), accuracies)
    plt.show()


def get_accuracy_distribution():
    event = get_latest_event_id()
    data = load_nas_data()
    accuracies = [x[1]*100. for x in data]
    accuracies = [int(x) for x in accuracies]
    sorted_accs = np.sort(accuracies)
    count_dict = {k: len(list(v)) for k, v in groupby(sorted_accs)}
    plt.bar(list(count_dict.keys()), list(count_dict.values()))
    plt.show()


########################################################
#                   DATASET LOADING                    #
########################################################
"""
# MNIST dataset
def load_dataset(path='../DATASETS'):

    # You can change this to include augmention and normalization

    valid_split_rate = VALIDATION_SPLIT_RATE
    error_msg = "[!] valid_split_rate should be in the range [0, 1]."
    assert ((valid_split_rate >= 0) and (valid_split_rate <= 1)), error_msg

    # Declare transform to convert raw data to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Loading Data and splitting it into train and validation data
    train = datasets.MNIST(root = path, train = True, transform = transform, download = True)
    train, valid = random_split(train,[int(len(train) - (valid_split_rate * len(train))), int(valid_split_rate * len(train))]) #, generator=Generator().manual_seed(MANUAL_SEED))
    test = datasets.MNIST(root = path, train = False, transform = transform, download = True)
    
    # Create Dataloader of the above tensor with batch size defined in "CONSTANTS.py"
    train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = DataLoader(valid, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)

    return train_loader, valid_loader, test_loader
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# In CONSTANTS.py, add a new constant for the dataset choice
# DATASET_CHOICE = 'Iris'  # or 'MNIST' or 'CIFAR-10'

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if DATASET_CHOICE == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET_CHOICE == 'CIFAR-10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif DATASET_CHOICE == 'Fashion-MNIST':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Invalid dataset choice')

    # Split the training set into a smaller training set and a validation set
    # Here, we use 80% of the data for training and 20% for validation
    num_train = len(trainset)
    num_val = int(0.2 * num_train)
    num_train = num_train - num_val
    trainset, valset = random_split(trainset, [num_train, num_val])

    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader
