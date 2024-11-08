import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    filename='logs/stock_prediction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StockDataset(Dataset):
    """Dataset for stock data without sentiment"""
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.features = data[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']].values
        self.labels = data['movement'].values

    def __len__(self):
        return len(self.data) - self.seq_length  # Return data length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]  # Features from idx to idx+seq_length
        y = self.labels[idx+self.seq_length]  # Label for the next day
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class StockDatasetWithSentiment(Dataset):
    """Dataset for stock data with sentiment"""
    def __init__(self, data, seq_length, sentiment_data):
        self.data = data
        self.seq_length = seq_length
        self.features = data[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']].values
        self.labels = data['movement'].values
        self.sentiment_data = sentiment_data[['positive', 'neutral', 'negative']].values

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_length]
        y = self.labels[idx+self.seq_length]
        sentiment = self.sentiment_data[idx+self.seq_length]

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(sentiment, dtype=torch.float32))

class BaseRNN(nn.Module):
    """Base RNN model without sentiment"""
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BaseRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, output_size), nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class StockRNN(nn.Module):
    """RNN model with sentiment"""
    def __init__(self, input_size, hidden_size, output_size, sentiment_size, num_layers):
        super(StockRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + sentiment_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, sentiment_data):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        _, hn = self.rnn(x, h0)
        sentiment_data = sentiment_data.view(sentiment_data.size(0), -1)
        combined = torch.cat((hn[-1], sentiment_data), dim=1)
        return self.fc(combined)

def preprocess_stock_data(data):
    """Preprocess stock data with normalization"""
    scaler = StandardScaler()
    data = data.copy()
    
    # Calculate price movement (1 for increase, 0 for decrease)
    data['Prev Adj'] = data['Adj Close'].shift(1)
    data['movement'] = data.apply(
        lambda row: 1 if row['Adj Close'] > row['Prev Adj'] else 0,
        axis=1
    )
    
    # Normalize features
    columns_to_scale = ['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    # Remove first row (has NaN due to shift) and drop temporary column
    data = data.iloc[1:].copy()
    data.drop(columns=['Prev Adj'], errors='ignore', inplace=True)
    
    return data

def evaluate_model(loader, model, with_sentiment=False):
    """Evaluate model performance"""
    model.eval()
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for batch in loader:
            if with_sentiment:
                features, labels, sentiment = batch
                outputs = model(features, sentiment)
            else:
                features, labels = batch
                outputs = model(features)
                
            _, predicted = torch.max(outputs, 1)
            
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    macc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return {
        'accuracy': accuracy,
        'mcc': macc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def train_model(model, train_loader, num_epochs, criterion, optimizer, with_sentiment=False):
    """Train the model"""
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            if with_sentiment:
                features, labels, sentiment = batch
                outputs = model(features, sentiment)
            else:
                features, labels = batch
                outputs = model(features)
                
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses

def main():
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    stock_data = pd.read_csv('data/raw/AAPL.csv')
    sentiment_data = pd.read_csv('data/processed/sentiment_AAPL.csv')
    
    # Preprocess data
    processed_data = preprocess_stock_data(stock_data)
    
    # Split data
    train_data = processed_data[
        (processed_data['Date'] >= '2014-01-01') & 
        (processed_data['Date'] < '2015-10-01')
    ]
    test_data = processed_data[
        (processed_data['Date'] >= '2015-10-01') & 
        (processed_data['Date'] < '2016-01-01')
    ]
    
    # Train models and evaluate
    models = {
        'BaseRNN': {'with_sentiment': False},
        'StockRNN': {'with_sentiment': True}
    }
    
    results = {}
    for model_name, config in models.items():
        logging.info(f"Training {model_name}")
        # Model training code here...

if __name__ == "__main__":
    main()