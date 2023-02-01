import os

# Mounting images directories
BASE_IN = "C:\\Users\\Alessio\\Desktop\\Archive"
TRAIN_DIR = os.path.sep.join([BASE_IN, "Train"])
TEST_DIR = os.path.sep.join([BASE_IN + "Test"])
ANNOTATION_TRAIN = os.path.sep.join([BASE_IN, "Train.csv"])
ANNOTATION_TEST = os.path.sep.join([BASE_IN, "Test.csv"])

#Mounting output directories
BASE_OUT = os.path.sep.join([BASE_IN, "Output"])
MODEL_PATH = os.path.sep.join([BASE_OUT, "model.h5"])
LB_PATH = os.path.sep.join([BASE_OUT, "lb.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUT, "plots"])

# Fixed variables
WIDTH = 224
HEIGHT = 224
N_CLASSES = 43

#Deep learning hyperparameters
BATCH_SIZE = 32
N_CHANNELS = 3
INIT_LR = 1e-4
NUM_EPOCHS = 20

