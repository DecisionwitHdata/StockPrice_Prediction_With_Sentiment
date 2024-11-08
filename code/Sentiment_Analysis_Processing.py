import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords
import torch
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import logging

# Set up logging
logging.basicConfig(
    filename='logs/sentiment_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SentimentAnalysis:
    """Class for analyzing sentiment in financial tweets using FinBERT"""
    
    def __init__(self, directory, price_data):
        """
        Initialize sentiment analysis for a company
        Args:
            directory (str): Directory containing tweet data
            price_data (pd.DataFrame): Price data for all companies
        """
        self.directory = directory
        self.company_name = os.path.basename(directory)
        self.price_data = price_data
        self.df = self.load_data()
        
        # Initialize NLTK resources
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.update({'rt', 'URL', 'AT_USER'})
        self.remove_chars = ['$']
        
        logging.info(f"Initialized sentiment analysis for {self.company_name}")

    def load_data(self):
        """Load and preprocess tweet data"""
        logging.info(f"Loading data for {self.company_name}")
        data = {'filename': [], 'sentences': []}
        
        for filename in os.listdir(self.directory):
            filepath = os.path.join(self.directory, filename)
            with open(filepath, 'r') as file:
                sentences = set(file.readlines())
                data['filename'].append(filename)
                data['sentences'].append(sentences)
                
        df = pd.DataFrame(data)
        df['preprocessed_text'] = df['sentences'].apply(self.preprocess_data)
        df['cleaned_tweet'] = df['preprocessed_text'].apply(self.clean_tweet)
        df = df.drop(columns=['sentences', 'preprocessed_text'])
        
        return df

    def preprocess_data(self, data_set):
        """Preprocess raw tweet data"""
        combined_text = []
        for json_obj in data_set:
            try:
                item = json.loads(json_obj)
                if 'text' in item:
                    combined_text.extend(item['text'])
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse JSON in {self.company_name}")
                continue
        return combined_text

    def clean_tweet(self, words):
        """Clean tweet text by removing stopwords and special characters"""
        return [
            word for word in words
            if word.lower() not in self.stopwords 
            and not any(char in word for char in self.remove_chars)
        ]

    def filter_price_data(self):
        """Filter price data for the specific company"""
        company_price_data = self.price_data[
            self.price_data['Stock'] == self.company_name
        ][['Date']]
        company_price_data['Date'] = pd.to_datetime(company_price_data['Date'])
        return company_price_data

    def fill_missing_dates(self):
        """Fill missing dates with neutral sentiment"""
        price_data = self.filter_price_data()
        
        # Prepare date columns
        df1 = pd.DataFrame({'date': pd.to_datetime(price_data['Date'])})
        df2 = pd.DataFrame({'date': pd.to_datetime(self.df['filename'])})
        
        df1.set_index('date', inplace=True)
        df2.set_index('date', inplace=True)
        
        # Find and fill missing dates
        missing_dates = df1.index.difference(df2.index)
        missing_df = pd.DataFrame({
            'filename': pd.to_datetime(missing_dates),
            'cleaned_tweet': [[''] for _ in missing_dates]
        })
        
        # Combine and sort data
        self.df = pd.concat([self.df, missing_df], ignore_index=True)
        self.df['filename'] = pd.to_datetime(self.df['filename'])
        self.df = self.df.sort_values(by='filename').reset_index(drop=True)
        
        # Keep only dates present in price data
        self.df = self.df[self.df['filename'].isin(df1.index)]
        
        logging.info(f"Filled {len(missing_dates)} missing dates for {self.company_name}")

    def analyze_sentiment(self):
        """Analyze sentiment using FinBERT"""
        logging.info(f"Starting sentiment analysis for {self.company_name}")
        
        # Initialize FinBERT
        finbert = BertForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone', 
            num_labels=3
        )
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        nlp = pipeline(
            "text-classification", 
            model=finbert, 
            tokenizer=tokenizer, 
            return_all_scores=True
        )

        def get_sentiment_for_text(text):
            """Get sentiment scores for a single text"""
            text = str(text)
            tokens = tokenizer.tokenize(text)[:510]
            text = tokenizer.convert_tokens_to_string(tokens)
            
            result = nlp(text)
            scores = {item['label'].lower(): item['score'] for item in result[0]}
            
            return {
                'label': max(result[0], key=lambda x: x['score'])['label'],
                'positive': scores.get('positive', 0),
                'negative': scores.get('negative', 0),
                'neutral': scores.get('neutral', 0),
                'sentiment_score': scores.get('positive', 0) - scores.get('negative', 0)
            }

        # Apply sentiment analysis
        self.df[['label', 'positive', 'negative', 'neutral', 'sentiment_score']] = (
            self.df['cleaned_tweet'].apply(get_sentiment_for_text).apply(pd.Series)
        )
        
        # Set neutral scores for missing tweets
        empty_tweets = self.df['cleaned_tweet'].apply(lambda x: x == [''])
        self.df.loc[empty_tweets, ['positive', 'negative', 'neutral']] = 0.333333
        self.df.loc[empty_tweets, 'sentiment_score'] = 0.0
        
        logging.info(f"Completed sentiment analysis for {self.company_name}")

def save_combined_to_csv(all_data, output_path):
    """Save combined sentiment data to CSV"""
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(output_path, index=False)
    logging.info(f"Saved combined data to {output_path}")

def run_analysis(base_directory, price_file, output_file):
    """Run sentiment analysis for all companies"""
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load price data
    price_data = pd.read_csv(price_file)
    all_data = []

    for company in os.listdir(base_directory):
        company_dir = os.path.join(base_directory, company)
        if os.path.isdir(company_dir):
            try:
                sa = SentimentAnalysis(company_dir, price_data)
                sa.fill_missing_dates()
                sa.analyze_sentiment()
                sa.df['company'] = company
                all_data.append(sa.df)
                logging.info(f"Completed analysis for {company}")
            except Exception as e:
                logging.error(f"Error processing {company}: {str(e)}")

    save_combined_to_csv(all_data, output_file)

if __name__ == "__main__":
    # Use relative paths
    base_directory = "data/raw/tweets"
    price_file = "data/processed/price_scaled.csv"
    output_file = "data/processed/all_companies_sentiment.csv"
    
    run_analysis(base_directory, price_file, output_file)