from utils import *
from mlpnas import MLPNAS 
from CONSTANTS import TOP_N


train_loader, valid_loader, test_loader = load_dataset()
data_loader = (train_loader, valid_loader)

nas_object = MLPNAS(data_loader)
data = nas_object.search()

# get_top_n_architectures(TOP_N)
