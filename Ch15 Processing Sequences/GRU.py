import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim import SGD, Adam
import numpy as np

class GRU(nn.Module):
    def __init__(self, n_feats: int = 1, n_hidden: int = 1, n_layers: int = 1, 
                 batch_first: bool = True, dropout: float = 0.0, 
                 bidir: bool = False, n_outputs: int = 1):
        super().__init__()
        self.gru_layer = nn.GRU(input_size = n_feats, hidden_size = n_hidden,
                                num_layers = n_layers, batch_first = batch_first,
                                dropout = dropout, bidirectional = bidir)
        self.linear = nn.Linear(n_hidden, n_outputs)
        
    def forward(self, X):
        _, h_n = self.gru_layer(X)
        return self.linear(h_n)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            _, h_n = self.gru_layer(X)
            return self.linear(h_n)
    
    def fit(self, train_dataset: TensorDataset, val_dataset: TensorDataset, batch_size: int = 64, 
            learning_rate: float = 1e-3, num_epochs: int = 100, device: str = 'cpu'):
        
        loss_fn = nn.MSELoss()
        optimizer = Adam(self.parameters(), lr = learning_rate)
        
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size,
                                      shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size,
                                    shuffle = True)
        print('Epoch\t Train Loss\t Val Loss\t Val RMSE')
        for epoch in range(num_epochs):
            train_loss = self.__train(train_dataloader, optimizer, loss_fn,
                                      device)
            val_loss, val_rmse = self.__test(val_dataloader, loss_fn, device)
            
            print(f'{epoch+1}\t {train_loss:.3f}\t {val_loss:.3f}\t {val_rmse:.2f}')
            
    def __train(self, train_dataloader, optimizer, loss_fn, device):
        train_loss = 0
        self.train()  
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            preds = self(X)
            preds = torch.squeeze(preds)[-1,:]
            y = torch.squeeze(y)
            loss = loss_fn(preds, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            
        return train_loss
    
    def __test(self, val_dataloader, loss_fn, device):
        sq_err = 0
        val_loss = 0
        self.eval()
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_dataloader):
                X, y = X.to(device), y.to(device)
                preds = self(X)
                preds = torch.squeeze(preds)[-1,:]
                y = torch.squeeze(y)
                loss = loss_fn(preds, y)
                val_loss += loss.item()
                sq_err += torch.sum((preds - y)**2).item()
        
        rmse = np.sqrt(sq_err/len(val_dataloader.dataset))
        
        return val_loss, rmse