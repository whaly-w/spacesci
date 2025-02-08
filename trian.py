import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from model import LSTMModel, AttentionLSTM
import argparse

###--------------------------- Setup ---------------------------
## Get extra argument
parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir', default='results', help='directory for saving trianed mode')
parser.add_argument('--model', default='LSTM', help='LSTM, BuffedLSTM, AttentionLSTM')
parser.add_argument('--dataset', default=10, help='dataset size [10y, 20y]')
args = parser.parse_args()

## Set directory for saved model
directory = args.dir

## Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



###--------------------------- Data Preparation ---------------------------
## Read data from CSV
dataset_size = args.dataset
doc = pd.read_csv('./datasets/dataset.csv') if dataset_size == 10 else pd.read_csv('./datasets_20y/dataset.csv')

## Drop out timestamp
doc = doc.drop(columns= 'datatime')
doc.head()

## Convert to np array
data = doc.to_numpy()
# print(data)

## Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
# print(normalized_data)

## Create data sequence
x, y = [], []
input_length = 24
output_length = 6
for i in range(len(data) - input_length - output_length + 1):
    x.append(data[i:i+input_length])  # Input sequence (24 time steps)
    y.append(data[i+input_length : i+input_length+output_length, 2]) 
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)

## Convert data to PyTorch tensor
x_ten = torch.tensor(x, dtype= torch.float32).to(device)
y_ten = torch.tensor(y, dtype= torch.float32).to(device)

## Split the data
x_train, x_val, y_train, y_val = train_test_split(x_ten, y_ten, test_size= 0.2, shuffle= False)
print(f'Train:\t{x_train.shape}, {y_train.shape}\nVal:\t{x_val.shape}, {y_val.shape}')

## Create batches for train & val
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size= 32, shuffle= False)


###--------------------------- Model Preparation ---------------------------
## Tuning Parameters
n_hidden = 2  # Number of LSTM hidden units
n_lstm_layers = 1  # Number of LSTM layers
learning_rate = 0.001
num_epochs = 10000
batch_size = 32

if __name__ == '__main__':
    ## Initialize the model, loss function, and optimizer
    model_selection = args.model
    if model_selection == 'LSTM':
        model = LSTMModel(n_hidden, n_lstm_layers).to(device)
    elif model_selection == 'BuffedLSTM':
        model = LSTMModel(5, 3).to(device)
    elif model_selection == 'AttentionLSTM':
        model = AttentionLSTM(n_hidden, n_lstm_layers).to(device)
    
    print(f'\n------------- Training {model_selection} with {dataset_size}y dataset -------------')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    ## Training loop
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(num_epochs):
        ## Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        ## Calculate average training loss
        train_loss /= len(train_loader)
        
        
        ## Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        ## Calculate average validation loss
        val_loss /= len(val_loader)
        
        ## Print training and validation loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        ## Save the model if validation loss improves
        if val_loss < best_val_loss:
            print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...')
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            ## Save the model checkpoint
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }
            torch.save(checkpoint, f'./{directory}/model{round(best_val_loss*10000)}.pth')
            torch.save(checkpoint, f'./{directory}/best_model.pth')
            
    print('Training Complete...')
    print(f'Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
