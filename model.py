import pandas as pd
import numpy as np
import torch
import torch.nn as nn

###--------------------------- Model Preparation ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_hidden, n_lstm_layers= 1):
        super(LSTMModel, self).__init__()
        self.n_hidden = n_hidden
        self.n_lstm_layers = n_lstm_layers
        
        self.lstm = nn.LSTM(3, n_hidden, n_lstm_layers, batch_first= True)
        self.fc = nn.Linear(self.n_hidden, 6)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    
class AttentionLSTM(nn.Module):
    def __init__(self, n_hidden, n_lstm_layers=1):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(3, n_hidden, n_lstm_layers, batch_first=True)
        self.attention = nn.Linear(n_hidden, n_hidden)  # Attention mechanism
        self.fc = nn.Linear(n_hidden, 6)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Attention mechanism
        attention_weights = nn.functional.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, hidden_size)
        attention_applied = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)

        out = self.fc(attention_applied)  # (batch_size, output_size)
        return out
    

class CNNLSTMModel(nn.Module):
    def __init__(self, n_cnn_hidden, n_lstm_hidden, n_lstm_layers=1, kernel_size=3, stride=1, dropout=0.2, n_input= 3, n_output= 6):
        super(CNNLSTMModel, self).__init__()
        self.n_hidden = {
            'n_lstm_hidden': n_lstm_hidden,
            'n_cnn_hidden_layer': n_cnn_hidden
        }
        self.n_lstm_layers = n_lstm_layers
        
        # 1D CNN layer
        self.cnn = nn.Conv1d(
            in_channels= n_input,  # Number of input features (a, b, c)
            out_channels= n_cnn_hidden,  # Number of output channels (same as LSTM hidden size)
            kernel_size= kernel_size,  # Size of the convolutional kernel
            stride= stride,  # Stride of the convolution
            padding= (kernel_size - 1) // 2  # Padding to maintain sequence length
        )
        
        # Max-pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduces sequence length by half
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_cnn_hidden,  # Input size to LSTM (same as CNN output channels)
            hidden_size=n_lstm_hidden,  # Number of LSTM hidden units
            num_layers=n_lstm_layers,  # Number of LSTM layers
            batch_first=True,  # Input shape: (batch_size, seq_length, input_size)
            dropout=dropout if n_lstm_layers > 1 else 0  # Dropout for multi-layer LSTM
        )
        
        # Add activatino funciton
        self.tanH = nn.Tanh()
        
        # Batch normalizer
        self.batch_norm = nn.BatchNorm1d(n_lstm_hidden)
        
        # Fully connected layer
        self.fc = nn.Linear(n_lstm_hidden, n_output)  # Output size is 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Permute input for CNN: (batch_size, input_size, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply CNN &  max-pooling
        x = self.cnn(x)
        x = self.pool(x)
        
        # Permute back for LSTM: (batch_size, seq_length // 2, hidden_size)
        x = x.permute(0, 2, 1)
        
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)
        
        # Apply LSTM
        out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_length // 2, hidden_size)
        
        # Apply activation function
        # out = self.tanH(out[:, -1, :])
        
        # Apply batch normalizer before DENSE layer
        out = self.batch_norm(out[:, -1, :])
        
        # Use the output of the last time step
        out = self.fc(out)  # Output shape: (batch_size, output_size)
        
        return out


class AutoregLSTM(nn.Module):
    def __init__(self, n_cnn_hidden, n_lstm_hidden, n_lstm_layers=2, kernel_size=3, stride=1, dropout=0.2, n_input= 3, n_output= 1):
        super(AutoregLSTM, self).__init__()
        self.n_hidden = {
            'n_lstm_hidden': n_lstm_hidden,
            'n_cnn_hidden_layer': n_cnn_hidden
        }
        self.n_lstm_layers = n_lstm_layers
        
        # 1D CNN layer
        self.cnn = nn.Conv1d(
            in_channels= n_input,  # Number of input features (a, b, c)
            out_channels= n_cnn_hidden,  # Number of output channels (same as LSTM hidden size)
            kernel_size= kernel_size,  # Size of the convolutional kernel
            stride= stride,  # Stride of the convolution
            padding= (kernel_size - 1) // 2  # Padding to maintain sequence length
        )
        
        # Max-pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Reduces sequence length by half
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_cnn_hidden,  # Input size to LSTM (same as CNN output channels)
            hidden_size=n_lstm_hidden,  # Number of LSTM hidden units
            num_layers=n_lstm_layers,  # Number of LSTM layers
            batch_first=True,  # Input shape: (batch_size, seq_length, input_size)
            dropout=dropout if n_lstm_layers > 1 else 0  # Dropout for multi-layer LSTM
        )
        
        # Batch normalizer
        self.batch_norm = nn.BatchNorm1d(n_lstm_hidden)
        
        # Linear projection
        self.proj = nn.Linear(n_lstm_hidden, n_cnn_hidden) 
        
        # Fully connected layer
        self.fc = nn.Linear(n_lstm_hidden, n_output)  # Output size is 1
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Permute input for CNN: (batch_size, input_size, seq_length)
        x = x.permute(0, 2, 1)
        
        # Apply CNN &  max-pooling
        x = self.cnn(x)
        x = self.pool(x)
        
        # Permute back for LSTM: (batch_size, seq_length // 2, hidden_size)
        x = x.permute(0, 2, 1)
        
        res_list = []
        for n in range(6):
            # Initialize hidden state and cell state for LSTM
            h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)
            c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)
            
            # Apply LSTM
            out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_length // 2, hidden_size)

            # Project out to shape of x for regression
            x = self.proj(out)

            # Apply batch normalizer before DENSE layer
            out = self.batch_norm(out[:, -1, :])
            
            
            out = self.fc(out)  # Output shape: (batch_size, output_size)
            res_list.append(out)
        
        return torch.cat(res_list, dim= 1)
    


class CNNLSTMModelV2(nn.Module):
    def __init__(self, n_cnn_hidden, n_lstm_hidden, n_lstm_layers=1, kernel_size=3, stride=1, dropout=0.2, n_input=3, n_output=6):
        super(CNNLSTMModelV2, self).__init__()
        self.n_hidden = {
            'n_lstm_hidden': n_lstm_hidden,
            'n_cnn_hidden_layer': n_cnn_hidden
        }
        self.n_lstm_layers = n_lstm_layers

        # LSTM layer (now processes the raw input)
        self.lstm = nn.LSTM(
            input_size=n_input,  # Input size is now the number of features (3)
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0
        )

        # Linear layer to project LSTM output to CNN input channels
        self.projection = nn.Linear(n_lstm_hidden, n_cnn_hidden)

        # 1D CNN layer (now processes LSTM output)
        self.cnn = nn.Conv1d(
            in_channels=n_cnn_hidden,  # Input channels = LSTM hidden size
            out_channels=n_cnn_hidden,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2
        )

        # Max-pooling layer (optional, but often useful)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.batch_norm = nn.BatchNorm1d(n_cnn_hidden)

        # Fully connected layer
        self.fc = nn.Linear(n_cnn_hidden, n_output)  # Output size remains the same
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Initialize hidden state and cell state for LSTM
        h0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)
        c0 = torch.zeros(self.n_lstm_layers, x.size(0), self.n_hidden['n_lstm_hidden']).to(self.device)

        # Apply LSTM
        out, _ = self.lstm(x, (h0, c0))  # Output shape: (batch_size, seq_length, hidden_size)

        # Project LSTM output to match CNN input channels
        out = self.projection(out)

        # Permute for CNN: (batch_size, channels, seq_length)
        out = out.permute(0, 2, 1)

        # Apply CNN & Max Pooling
        out = self.cnn(out)
        out = self.pool(out)  # Optional max pooling

        # Apply batch normalizer
        out = self.batch_norm(out)

        # Flatten for the fully connected layer
        out = out.permute(0, 2, 1).reshape(out.size(0), -1)

        # Fully connected layer
        out = self.fc(out)  # Output shape: (batch_size, n_output)

        return out
