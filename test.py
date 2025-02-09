import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import LSTMModel, CNNLSTMModel
import argparse

###--------------------------- Setup ---------------------------
## Get extra argument
parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir_data', required= True, help='directory for the test data')
parser.add_argument('--dir_model', required= True, help='directory for the saved model')
parser.add_argument('--model', default='LSTM', help='LSTM, BuffedLSTM, CNNLSTM')

args = parser.parse_args()

## Set parameter for the model
directory = args.dir_data       # directory for the test data
model_dir = f'./{args.dir_model}/best_model.pth'      # directory for the saved model
model_selection = args.model    # choose model

## Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



###--------------------------- Data Preparation ---------------------------
## Read data from CSV
doc = pd.read_csv(f'./{directory}/dataset.csv')

## Drop out timestamp
doc = doc.drop(columns= 'datatime')
doc.head()

## Convert to np array
data = doc.to_numpy()

## Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

y = [[i[2]] for i in data]
y_scaler = MinMaxScaler()
# print(y)
_ = y_scaler.fit_transform(y)


## Create data sequence using sliding widow technique
x, y = [], []
input_length = 24
output_length = 6
for i in range(len(normalized_data) - input_length - output_length + 1):
    x.append(normalized_data[i:i+input_length])
    y.append(normalized_data[i+input_length : i+input_length+output_length, 2]) 
        
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

## Convert data to PyTorch tensor
x_ten = torch.tensor(x, dtype= torch.float32).to(device)
y_ten = torch.tensor(y, dtype= torch.float32).to(device)


###--------------------------- Model Preparation ---------------------------
if __name__ == '__main__':
    print(f'\n------------- Using {model_selection} with {len(x_ten)} test data -------------')
    ## Initialize the model, loss function, and optimizer
    if model_selection == 'LSTM':
        model = LSTMModel(2, 1).to(device)
    elif model_selection == 'BuffedLSTM':
        model = LSTMModel(5, 3).to(device)
    elif model_selection == 'CNNLSTM':
        model = CNNLSTMModel(32, 4).to(device)
        
    ## Load checkpoint
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    model.eval()
    
    for i in range(len(x_ten)):
        # x_test = torch.tensor([x_ten[i]], dtype= torch.float32).to(device)
        test_input = x_ten[-1].unsqueeze(0).to(device)
        y_pred = model(test_input)
        y_target = y_ten[i]

        y_pred_scaler = y_pred.cpu().detach().numpy().reshape(-1, 1)
        y_pred_denomalized = y_scaler.inverse_transform(y_pred_scaler)
        
        y_target_scaler = y_target.cpu().detach().numpy().reshape(-1, 1)
        y_target_denomalized = y_scaler.inverse_transform(y_target_scaler)
        
        print(f'Iteration {i} -\tMSE Loss: {mean_squared_error(y_target_denomalized, y_pred_denomalized):.3f}, Norm MSE Loss: {mean_squared_error(y_target_scaler, y_pred_scaler):.3f}')
        print(y_pred_denomalized)
    