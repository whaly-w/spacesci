import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from model import AutoregLSTM, CNNLSTMModel
import argparse
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter

###--------------------------- Setup ---------------------------
## Get extra argument
parser = argparse.ArgumentParser(description='Add these argument for training')
parser.add_argument('--dir', default='results', help='directory for saving trianed mode')

parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--epochs', default=1000, help='epoch number')
parser.add_argument('--batch_size', default=32)
args = parser.parse_args()

## Set parameter for the model
directory = args.dir                # directory for saving model
batch_size = int(args.batch_size)   # batch size
learning_rate = float(args.lr)
num_epochs = int(args.epochs)

## Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



###--------------------------- Data Preparation ---------------------------
## Read data from CSV
doc = pd.read_csv('./datasets_20y/dataset.csv')

## Drop out timestamp
doc = doc.drop(columns= 'datatime')
doc = doc.drop(columns= 'Kp')
doc = doc.drop(columns= 'Hp60')
print(doc.head())


## Convert to np array
data = doc.to_numpy()
# print(data)

## Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)
# print(normalized_data)

## Create data sequence using sliding widow technique
x, y = [], []
input_length = 24
output_length = 6
for i in range(len(normalized_data) - input_length - output_length + 1):
    x.append(normalized_data[i:i+input_length])
    y.append(normalized_data[i+input_length : i+input_length+output_length, -1]) 
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
print(y)

## Convert data to PyTorch tensor
x_ten = torch.tensor(x, dtype= torch.float32).to(device)
y_ten = torch.tensor(y, dtype= torch.float32).to(device)

## Split the data
x_train, x_val, y_train, y_val = train_test_split(x_ten, y_ten, test_size= 0.2, shuffle= False)
print(f'Train:\t{x_train.shape}, {y_train.shape}\nVal:\t{x_val.shape}, {y_val.shape}')

## Create batches for train & val
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle= False)


###--------------------------- Model Preparation ---------------------------
if __name__ == '__main__':
    ## Initialize the model, loss function, and optimizer
    model = AutoregLSTM(64, 48).to(device)
        
    
    print(f'\n------------- Training with lr: {learning_rate}, batch size: {batch_size}, n_epochs: {num_epochs} -------------')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    

    ## Training loop
    best_val_loss = 1e9
    best_epoch = 0
    doc = []
    for epoch in range(num_epochs):
        ## Training phase
        model.train()
        train_loss = 0.0
        for i, (batch_X, batch_y) in enumerate(train_loader):
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
             # Apply gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
            
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
            
        ## Record value
        doc.append({
            'epochs': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    
    ## Write .csv file
    df = pd.DataFrame(doc)    
    df.to_csv(f'{directory}/train.csv', index= False)

    print('Training Complete...')
    print(f'Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
    
    
