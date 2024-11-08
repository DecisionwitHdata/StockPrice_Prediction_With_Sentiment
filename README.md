# Stock Market Prediction Project

This project aims to predict stock price movements by analyzing historical price data and sentiment data extracted from Twitter. 
We leveraged FinBERT, a financial domain-specific NLP model, to evaluate if public sentiment influences stock trends.

## Project Overview

Using data from **StockNet** and Twitter, this project combines daily stock prices with sentiment analysis to predict stock movements. 
The project examines models trained on price data alone and in combination with sentiment data.

## Data

### 1. Stock Price Data
We used stock price data from StockNet, covering 88 companies in sectors like Consumer Goods and Healthcare, spanning January 1, 2014, to January 1, 2016. 
The data includes columns for date, open, high, low, close, adjusted close, and volume.

### 2. Sentiment Data
Twitter data was retrieved using NASDAQ ticker symbols (e.g., `$GOOG` for Google). Sentiment analysis was conducted using FinBERT, which labeled each tweet as positive, neutral, or negative. Missing sentiment scores were filled with a neutral score (0.333 each for positive, neutral, and negative).

## Model Description

We used FinBERT for sentiment analysis, which performed better than general models on financial text tasks. 
For price movement prediction, we trained baseline RNN and LSTM models on price data alone, and then tested hybrid models combining price and sentiment data.

## Experiment Setup

### 1. Data Preprocessing
- **Price Data**: A `movement` column was added to indicate daily price changes (1 for increase, 0 for decrease). The data was normalized using MinMaxScaler.
- **Sentiment Data**: Sentiment scores were normalized, and missing data was replaced with a neutral score.

### 2. Model Configurations and Parameters
- **Baseline Models (Price Only)**: RNN and LSTM models were trained on price data.
- **Hybrid Models (Price + Sentiment)**:
  - **StockRNN + Sentiment**
  - **StockRNN + SentimentRNN**

Below is the architecture diagram for the StockRNN + Senti model:

<img src="image/StockRNN_Senti.jpg" alt="StockRNN + Senti" width="800">
<img src="image/StockRNN_SentiRNN.jpg" alt="StockRNN+SentiRNN" width="800">

All models used the following parameters: batch size = 32, window size = 5, epochs = 120, hidden size = 32, layers = 1, learning rate = 0.001.

## Results

| Model                | MCC     |
|----------------------|---------|
| BaseRNN              | 0.0339  |
| BaseLSTM             | 0.0379  |
| StockRNN + Sentiment | 0.0503  |
| StockRNN + SentiRNN  | 0.0511  |

The results indicate that sentiment data slightly improved the MCC for certain models, but did not consistently enhance overall accuracy. 

## Conclusion

1. **Sentiment-Stock Correlation**: Sentiment analysis did not strongly correlate with stock price movements. Positive sentiment did not always lead to price increases, and negative sentiment did not consistently result in decreases.
  
2. **Neutral Sentiment Dominance**: Sentiment data was predominantly neutral, limiting its effectiveness in predicting movements. Future analyses could consider more granular sentiment scores.

3. **Model Complexity**: Simple RNN-based architectures struggled to capture the relationship between sentiment and price movement. Advanced models, such as attention-based architectures, may better leverage sentiment for stock prediction.

## Repository Structure

```plaintext
StockPrice_Prediction_With_Sentiment/
├── Code/
│   ├── Baseline_models.ipynb     # Base RNN/LSTM models without sentiment
│   ├── Main.ipynb               # Main experiment notebook
│   ├── Sentiment_Analysis_Preprocessing.py  # Sentiment analysis code
│   └── Stock_Data_Preprocessing.py          # Stock data preprocessing
│
├── Data/
│   ├── processed/               # Preprocessed data files
│   └── results/                 # Model results and evaluations
│
├── Experiments/
│   ├── AAPL_Sentiment.py        # Apple stock sentiment analysis
│   ├── AAPL_Stock_Prediction.py # Apple stock prediction models
│   ├── FB_GOOG_MSF.py          # Facebook, Google, Microsoft models
│   └── image/                   # Visualizations and plots
│
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
