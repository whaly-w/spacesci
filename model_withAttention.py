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