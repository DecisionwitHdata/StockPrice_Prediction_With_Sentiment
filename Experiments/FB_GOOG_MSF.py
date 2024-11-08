import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataProcessor:
    """Handle data loading and preprocessing"""
    def __init__(self, stock_file, sentiment_file=None, window_size=5):
        self.scaler = MinMaxScaler()
        self.window_size = window_size
        self.stock_file = stock_file
        self.sentiment_file = sentiment_file

    def load_data(self, selected_stocks=None):
        """Load and preprocess stock and sentiment data"""
        # Load stock data
        df = pd.read_csv(self.stock_file)
        if selected_stocks:
            df = df[df['Stock'].isin(selected_stocks)]

        # Preprocess stock data
        df.drop(columns='Prev Adj', errors='ignore', inplace=True)
        df[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']] = self.scaler.fit_transform(
            df[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']]
        )

        if self.sentiment_file:
            sentiment = pd.read_csv(self.sentiment_file)
            sentiment = self._preprocess_sentiment(sentiment)
            df = self._merge_stock_sentiment(df, sentiment)

        return df

    def _preprocess_sentiment(self, sentiment_df):
        """Preprocess sentiment data"""
        sentiment_df.drop(columns=['Unnamed: 0'], inplace=True)
        sentiment_df.rename(columns={'filename': 'Date'}, inplace=True)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        return sentiment_df

    def _merge_stock_sentiment(self, stock_df, sentiment_df):
        """Merge stock and sentiment data"""
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        return pd.merge(sentiment_df, stock_df, on=["Date", 'Stock'])

    def create_sequences(self, df, include_sentiment=False):
        """Create sequences for model input"""
        if include_sentiment:
            return self._create_sequences_with_sentiment(df)
        return self._create_sequences_price_only(df)

    def _create_sequences_price_only(self, df):
        """Create sequences for price-only data"""
        sequences = []
        labels = []

        for stock in df['Stock'].unique():
            stock_data = df[df['Stock'] == stock].drop(columns=['Stock', 'Date'])
            stock_data = stock_data.sort_index().to_numpy()

            for i in range(len(stock_data) - self.window_size):
                sequences.append(stock_data[i:i + self.window_size, :-2])
                labels.append(stock_data[i + self.window_size, -2])

        return np.array(sequences), np.array(labels)

    def _create_sequences_with_sentiment(self, df):
        """Create sequences including sentiment data"""
        total_sequences = []
        total_sentiments = []
        total_labels = []

        for stock in df['Stock'].unique():
            stock_data = df[df['Stock'] == stock].drop(
                columns=['Stock', 'Date', 'cleaned_tweet', 'label', 'Unnamed: 0']
            ).sort_index().to_numpy()

            for i in range(len(stock_data) - self.window_size):
                total_sequences.append(stock_data[i:i + self.window_size, 4:10])
                total_sentiments.append(stock_data[i:i + self.window_size, :3])
                total_labels.append(stock_data[i + self.window_size, -2])

        return np.array(total_sequences), np.array(total_sentiments), np.array(total_labels)

class StockDataset(Dataset):
    """Dataset for stock price prediction"""
    def __init__(self, sequences, labels, sentiments=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.sentiments = torch.FloatTensor(sentiments) if sentiments is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.sentiments is not None:
            return (self.sequences[idx], 
                    self.sentiments[idx], 
                    self.labels[idx])
        return self.sequences[idx], self.labels[idx]

class ModelTrainer:
    """Handle model training and evaluation"""
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, train_loader, num_epochs, valid_loader=None, with_sentiment=False):
        """Train the model"""
        train_losses = []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                loss = self._train_step(batch, with_sentiment)
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
                if valid_loader:
                    self.evaluate(valid_loader, with_sentiment)
        
        return train_losses

    def _train_step(self, batch, with_sentiment):
        """Perform one training step"""
        self.optimizer.zero_grad()
        
        if with_sentiment:
            sequences, sentiments, labels = [b.to(self.device) for b in batch]
            outputs = self.model(sequences, sentiments)
        else:
            sequences, labels = [b.to(self.device) for b in batch]
            outputs = self.model(sequences)
            
        labels = labels.squeeze(1).long()
        loss = self.criterion(outputs, labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss

    def evaluate(self, loader, with_sentiment=False):
        """Evaluate the model"""
        self.model.eval()
        tp = tn = fp = fn = 0
        
        with torch.no_grad():
            for batch in loader:
                metrics = self._evaluate_batch(batch, with_sentiment)
                tp += metrics['tp']
                tn += metrics['tn']
                fp += metrics['fp']
                fn += metrics['fn']

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator != 0 else 0

        return {
            'accuracy': accuracy,
            'mcc': mcc,
            'tp': tp, 'tn': tn,
            'fp': fp, 'fn': fn
        }

    def _evaluate_batch(self, batch, with_sentiment):
        """Evaluate one batch"""
        if with_sentiment:
            sequences, sentiments, labels = [b.to(self.device) for b in batch]
            outputs = self.model(sequences, sentiments)
        else:
            sequences, labels = [b.to(self.device) for b in batch]
            outputs = self.model(sequences)
            
        labels = labels.squeeze(1).long()
        _, predicted = torch.max(outputs, 1)
        
        return {
            'tp': ((predicted == 1) & (labels == 1)).sum().item(),
            'tn': ((predicted == 0) & (labels == 0)).sum().item(),
            'fp': ((predicted == 1) & (labels == 0)).sum().item(),
            'fn': ((predicted == 0) & (labels == 1)).sum().item()
        }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize data processor
    processor = DataProcessor(
        stock_file='data/processed/filtered_stock.csv',
        sentiment_file='data/processed/sentiment.csv'
    )
    
    # Load and process data
    selected_stocks = ['FB', 'GOOG', 'MSFT']
    df = processor.load_data(selected_stocks)
    
    # Create sequences
    sequences, labels = processor.create_sequences(df)
    
    # Create datasets and dataloaders
    # Training code here...

if __name__ == "__main__":
    main()