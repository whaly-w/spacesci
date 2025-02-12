import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from model import AutoregLSTM
import argparse

###--------------------------- Setup ---------------------------
## Get extra argument
parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir', required= True, help='directory for the test data')
parser.add_argument('--model', required= True, help='SN, F107')

parser.add_argument('--dir_model', default='results_final', help='directory for the saved model')

args = parser.parse_args()


## Set parameter for the model
directory = args.dir      # directory for the test data
model_selection = args.model    # choose model
model_dir = f'./{args.dir_model}/{model_selection}/best_model.pth'      # directory for the saved model

## Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


###--------------------------- Data Preparation ---------------------------
## Read data from CSV
doc = pd.read_csv(f'./{directory}/dataset.csv')

## Drop out timestamp
target_Hp60 = doc['Hp60']
doc = doc.drop(columns= ['datetime', 'Hp60'])
if model_selection == 'SN': doc = doc.drop(columns= 'F10.7')
elif model_selection == 'F107': doc = doc.drop(columns= 'SN')
print(doc.head())

## Convert to np array
data = doc.to_numpy()

## Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

y = [[i[-1]] for i in data]
y_scaler = MinMaxScaler()
_ = y_scaler.fit_transform(y)


## Create data sequence using sliding widow technique
x, y, target_norm = [], [], []
input_length = 24
output_length = 6
for i in range(len(normalized_data) - input_length - output_length + 1):
    x.append(normalized_data[i:i+input_length])
    y.append(target_Hp60[i+input_length : i+input_length+output_length])
    target_norm.append(normalized_data[i+input_length : i+input_length+output_length, -1])
        
x = np.array(x)
y = np.array(y)
target_norm = np.array(target_norm)
print('\ndata shape ->', x.shape, y.shape)

## Convert data to PyTorch tensor
x_ten = torch.tensor(x, dtype= torch.float32).to(device)


###--------------------------- Model Preparation ---------------------------
if __name__ == '__main__':
    print(f'\n------------- Testing with {len(x_ten)} blind test data -------------')
    ## Initialize the model, loss function, and optimizer
    model = AutoregLSTM(64, 48, n_input= 2).to(device)
        
    ## Load checkpoint
    model.load_state_dict(torch.load(model_dir, weights_only= False)['model_state_dict'])
    model.eval()
    
    for i in range(len(x_ten)):
        # x_test = torch.tensor([x_ten[i]], dtype= torch.float32).to(device)
        test_input = x_ten[i].unsqueeze(0).to(device)
        y_pred = model(test_input)
        y_target = y[i]
        y_target_normal = target_norm[i]
   
        # Denomalized data
        y_pred_scaler = y_pred.cpu().detach().numpy().reshape(-1, 1)
        y_pred_denomalized = y_scaler.inverse_transform(y_pred_scaler)
        
        # Map back to original data
        y_pred_origin = np.exp(y_pred_denomalized) - np.ones(y_pred_denomalized.shape)
            
        
        print(f'Case {i+1}\t| MSE Loss: {mean_squared_error(y_target, y_pred_origin.reshape(-1)):.3f}, Norm MSE Loss: {mean_squared_error(y_target_normal, y_pred_scaler.reshape(-1)):.3f}')
        print(f'\t| RMSE Loss: {root_mean_squared_error(y_target, y_pred_origin.reshape(-1)):.3f}, RNorm MSE Loss: {root_mean_squared_error(y_target_normal, y_pred_scaler.reshape(-1)):.3f}')
        print(f'target -> {y_target}')
        print(f'pred   -> {np.array([round(j, 3) for j in y_pred_origin.reshape(-1)])}\n')

    