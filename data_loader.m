% load data and feature vector in matlab
load('cuhk03_new_protocol_config_labeled.mat')
feature = jsondecode(fileread('feature_data.json'));