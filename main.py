from scipy.io import loadmat
import json

dir = "D:/EEE Year4/Representation-and-Distance-Metrics-Learning/PR_data/"
train_idxs = loadmat(dir + 'cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()

with open(dir + 'feature_data.json', 'r') as f:
    features = json.load(f)

