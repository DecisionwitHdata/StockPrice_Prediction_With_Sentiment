import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
import requests
from io import BytesIO

class StockDataPreprocessor:
    """
    Class for preprocessing stock market data and sentiment analysis
    """
    def __init__(self):
        # Create necessary directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def download_stock_data(self):
        """
        Download stock data from StockNet dataset
        """
        # Download ZIP file URL from GitHub repository
        url = 'https://github.com/yumoxu/stocknet-dataset/archive/refs/heads/master.zip'
        
        # Download ZIP file and read into memory
        response = requests.get(url)
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        
        # Folder path for 'stocknet-dataset-master/price/raw/' in ZIP file
        price_folder_path = 'stocknet-dataset-master/price/raw/'
        
        # Read and combine all CSV files
        data_frames = []
        for file_name in zip_file.namelist():
            if file_name.startswith(price_folder_path) and file_name.endswith('.csv'):
                # Extract stock name from filename (e.g., 'raw/prices/appl.csv' -> 'appl')
                stock_name = os.path.splitext(os.path.basename(file_name))[0]
                
                # Convert CSV file to DataFrame
                with zip_file.open(file_name) as file:
                    df = pd.read_csv(file)
                    # Add stock name column
                    df['Stock'] = stock_name
                    df['Date'] = pd.to_datetime(df['Date'])
                    data_frames.append(df)
        
        # Combine all DataFrames into one
        return pd.concat(data_frames, ignore_index=False)
    
    def preprocess_stock_data(self, combined_df):
        """
        Preprocess stock data by calculating movement and applying filters
        """
        # Calculate price movement and rate of change
        stocks = combined_df['Stock'].unique()
        for stock in stocks:
            combined_df['Prev Adj'] = combined_df['Adj Close'].shift(1)
            combined_df['movement'] = combined_df.apply(
                lambda row: 1 if row['Adj Close'] > row['Prev Adj'] else 0,
                axis=1
            )
            combined_df['Rate'] = combined_df['Adj Close'] / combined_df['Prev Adj'] - 1
            
        # Remove small price movements
        cleaned = combined_df[~((combined_df['Rate'] > -0.0055) & (combined_df['Rate'] <= 0.005))]
        
        # Filter date range
        filtered = cleaned[(cleaned['Date'] >= '2014-01-01') & (cleaned['Date'] <= '2016-01-01')]
        
        # Drop temporary column and normalize features
        filtered.drop(columns='Prev Adj', errors='ignore', inplace=True)
        filtered[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']] = self.scaler.fit_transform(
            filtered[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Close']]
        )
        
        return filtered
    
    def merge_sentiment_data(self, stock_data, sentiment_dir='data/processed/sentiment'):
        """
        Merge stock data with sentiment analysis data
        """
        stock_data = stock_data.copy()
        stock_data['date'] = pd.to_datetime(stock_data['Date'])
        
        sentiment_files = os.listdir(sentiment_dir)
        combined_sentiment_data = pd.DataFrame()
        
        for sentiment_file in sentiment_files:
            sentiment_file_path = os.path.join(sentiment_dir, sentiment_file)
            sentiment_data = pd.read_csv(sentiment_file_path)
            sentiment_data['filename'] = pd.to_datetime(sentiment_data['filename'])
            
            ticker = sentiment_file.split('_')[0]
            stock_data_ticker = stock_data[stock_data['Stock'] == ticker]
            
            merged_data = pd.merge(
                stock_data_ticker, 
                sentiment_data, 
                left_on='date', 
                right_on='filename', 
                how='left'
            )
            
            combined_sentiment_data = pd.concat([combined_sentiment_data, merged_data], ignore_index=True)
        
        # Fill missing sentiment scores with neutral values
        sentiment_columns = ['cleaned_tweet', 'label', 'positive', 'negative', 'neutral', 'sentiment_score']
        default_values = {
            'cleaned_tweet': '', 
            'label': '', 
            'positive': 0.33333, 
            'negative': 0.33333, 
            'neutral': 0.33333, 
            'sentiment_score': 0
        }
        combined_sentiment_data[sentiment_columns] = combined_sentiment_data[sentiment_columns].fillna(default_values)
        
        return combined_sentiment_data
    
    def process_and_save_data(self):
        """
        Main function to process and save all data
        """
        # Download and preprocess stock data
        combined_df = self.download_stock_data()
        filtered_data = self.preprocess_stock_data(combined_df)
        
        # Save preprocessed stock data
        filtered_data.to_csv('data/processed/filtered_stock.csv', index=False)
        
        # Merge with sentiment data
        final_data = self.merge_sentiment_data(filtered_data)
        
        # Save final merged data
        final_data.to_csv('data/processed/merged_stock_sentiment.csv', index=False)
        
        return final_data

def main():
    preprocessor = StockDataPreprocessor()
    final_data = preprocessor.process_and_save_data()
    print("Data preprocessing complete!")
    print(f"Final dataset shape: {final_data.shape}")

if __name__ == "__main__":
    main()