import pennylane as qml
import numpy as np
from src.data_preprocessing import data_preprocess
from config import train_config
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Load your configurations
filepath = train_config['training_dataset_path']
dr = train_config['dr_technique']
dr_comp = train_config['dr_components']
num_layers = train_config['training_layers']
ansatz = train_config['ansatz']
train_size = train_config['train_size']
alignment_epochs = train_config['alignment_epochs']

print('Reading the Data file ...')
try:
    x, y = data_preprocess(path=filepath, dr_type=dr, dr_components=dr_comp, normalize=False)
except Exception as e:
    print("Error while Reading the file")
    print(e)

num_qubits = len(x[0])

print("Creating Quantum Kernel Circuit...")

