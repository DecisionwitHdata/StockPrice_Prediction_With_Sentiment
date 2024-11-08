import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords
import torch
from transformers import BertForSequenceClassification, BertTokenizer, pipeline

# Create necessary directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

class SentimentAnalyzer:
    def __init__(self, company_code):
        """
        Initialize SentimentAnalyzer for a specific company
        Args:
            company_code (str): Company stock code (e.g., 'AAPL')
        """
        self.company_code = company_code
        self.directory = f'data/raw/{company_code}'
        self.price_data_path = f'data/raw/{company_code}.csv'
        self.output_path = f'data/processed/sentiment_{company_code}.csv'
        
        # Initialize FinBERT model
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("text-classification", model=self.finbert, tokenizer=self.tokenizer, return_all_scores=True)
        
        # Download NLTK stopwords
        nltk.download('stopwords')
        self.setup_stopwords()

    def setup_stopwords(self):
        """Setup stopwords for tweet cleaning"""
        nltk_stopwords = set(stopwords.words('english'))
        additional_stopwords = {'rt', 'URL', 'AT_USER'}
        self.all_stopwords = nltk_stopwords.union(additional_stopwords)
        self.remove_characters = ['$']

    def load_tweet_data(self):
        """Load and combine tweet data from files"""
        data = {'filename': [], 'sentences': []}
        
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            with open(filepath, 'r') as file:
                sentences = set(file.readlines())
                data['filename'].append(filename)
                data['sentences'].append(sentences)
                
        return pd.DataFrame(data)

    def preprocess_tweets(self, df):
        """Preprocess tweet data"""
        preprocessed_data = []
        for data_set in df['sentences']:
            combined_text = []
            for json_obj in data_set:
                item = json.loads(json_obj)
                if 'text' in item:
                    combined_text.extend(item['text'])
            preprocessed_data.append(combined_text)
            
        df['preprocessed_text'] = preprocessed_data
        return df

    def clean_tweet(self, words):
        """Clean tweet text by removing stopwords and special characters"""
        return [word for word in words
                if word.lower() not in self.all_stopwords 
                and not any(char in word for char in self.remove_characters)]

    def get_sentiment_for_text(self, text):
        """Analyze sentiment for text using FinBERT"""
        text = str(text)
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:
            tokens = tokens[:510]
        text = self.tokenizer.convert_tokens_to_string(tokens)

        result = self.nlp(text)
        
        # Extract probabilities
        positive_prob = next((item['score'] for item in result[0] if item['label'].lower() == 'positive'), 0)
        negative_prob = next((item['score'] for item in result[0] if item['label'].lower() == 'negative'), 0)
        neutral_prob = next((item['score'] for item in result[0] if item['label'].lower() == 'neutral'), 0)
        
        sentiment_score = positive_prob - negative_prob

        return {
            'label': max(result[0], key=lambda x: x['score'])['label'],
            'positive': positive_prob,
            'negative': negative_prob,
            'neutral': neutral_prob,
            'sentiment_score': sentiment_score
        }

    def process_sentiment(self):
        """Main processing function"""
        # Load and preprocess data
        df = self.load_tweet_data()
        df = self.preprocess_tweets(df)
        
        # Clean tweets
        df = df.drop(columns=['sentences', 'preprocessed_text'])
        
        # Set date as index and handle missing dates
        price_data = pd.read_csv(self.price_data_path)
        price_data = price_data[
            (price_data['Date'] >= '2014-01-01') & 
            (price_data['Date'] <= '2015-12-31')
        ]
        
        df['filename'] = pd.to_datetime(df['filename'])
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        
        # Find missing dates
        missing_dates = price_data['Date'].difference(df['filename'])
        
        # Add missing dates with neutral sentiment
        missing_df = pd.DataFrame({
            'filename': missing_dates,
            'cleaned_tweet': [[''] for _ in missing_dates]
        })
        
        df = pd.concat([df, missing_df], ignore_index=True)
        df = df.sort_values(by='filename').reset_index(drop=True)
        
        # Calculate sentiment
        df[['label', 'positive', 'negative', 'neutral', 'sentiment_score']] = (
            df['cleaned_tweet'].apply(self.get_sentiment_for_text).apply(pd.Series)
        )
        
        # Fill missing values with neutral sentiment
        mask = df['cleaned_tweet'].apply(lambda x: x == [''])
        df.loc[mask, ['positive', 'negative', 'neutral']] = 0.333333
        df.loc[mask, 'sentiment_score'] = 0.0
        
        # Save results
        df.to_csv(self.output_path, index=False)
        print(f"Processed {len(df)} tweets for {self.company_code}")
        
        return df

def main():
    # Example usage
    analyzer = SentimentAnalyzer('AAPL')
    sentiment_df = analyzer.process_sentiment()
    print("Sentiment analysis complete!")

if __name__ == "__main__":
    main()