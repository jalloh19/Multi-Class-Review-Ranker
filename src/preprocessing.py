"""
Text preprocessing module for Amazon review classification.
Includes POS-aware lemmatization for improved accuracy.
"""

import pandas as pd
import numpy as np
import re
import nltk
import logging

# Configure logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download required NLTK resources
def download_nltk_resources():
    """Download all required NLTK resources."""
    resources = ['stopwords', 'wordnet', 'words', 'punkt_tab', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            print(f"Warning: Could not download {resource}")

# Initialize resources
english_words = None
stop_words = None
lemmatizer = None

def load_resources():
    global english_words, stop_words, lemmatizer
    if english_words is not None:
        return

    download_nltk_resources()
    
    try:
        from nltk.corpus import words, wordnet, stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Ensure loaded
        try:
            words.ensure_loaded()
            wordnet.ensure_loaded()
        except AttributeError:
            pass

        english_words = set(words.words())
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                english_words.add(lemma.name().lower().replace('_', ' '))

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except Exception as e:
        logger.error(f"Failed to load NLTK resources: {e}")
        # Component-level fallback
        if english_words is None: english_words = set()
        if stop_words is None: stop_words = set()
        if lemmatizer is None: lemmatizer = WordNetLemmatizer()

# Initialize resources lazily in preprocess_text instead of top-level
# download_nltk_resources() <-- REMOVED BLOCKING CALL
# english_words = set(words.words()) <-- REMOVED BLOCKING CALL
# ...


def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags.
    
    Args:
        treebank_tag: POS tag from Penn Treebank
        
    Returns:
        WordNet POS tag
    Returns:
        WordNet POS tag (str)
    """
    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def preprocess_text(text, use_pos_tagging=True):
    """
    Clean and preprocess text with optional POS-aware lemmatization.
    
    Args:
        text: Raw text string
        use_pos_tagging: Whether to use POS tagging for better lemmatization
        
    Returns:
        Cleaned and preprocessed text string
    """
    # Helper to load on first use
    if english_words is None:
        load_resources()

    try:
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
    
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Keep only letters and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        if not tokens:
            return ""
        
        if use_pos_tagging:
            try:
                # POS-aware lemmatization
                from nltk import pos_tag
                pos_tags = pos_tag(tokens)
                final_tokens = []
                
                for word, tag in pos_tags:
                    # Filter stopwords and short words only (removed english_words check)
                    if (stop_words is None or word not in stop_words) and len(word) > 2:
                        wn_tag = get_wordnet_pos(tag)
                        if lemmatizer:
                            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
                        else:
                            lemma = word
                        final_tokens.append(lemma)
            except (LookupError, Exception):
                # Fallback: just filter stopwords
                final_tokens = [w for w in tokens 
                               if (stop_words is None or w not in stop_words) and len(w) > 2]
        else:
            # Simple lemmatization
            final_tokens = []
            for w in tokens:
                if (stop_words is None or w not in stop_words) and len(w) > 2:
                    if lemmatizer:
                        final_tokens.append(lemmatizer.lemmatize(w))
                    else:
                        final_tokens.append(w)
    
        return ' '.join(final_tokens)
    
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return ""


def load_and_clean_data(file_path, use_pos_tagging=True):
    """
    Load and preprocess the Amazon reviews dataset.
    
    Args:
        file_path: Path to the CSV file
        use_pos_tagging: Whether to use POS tagging
        
    Returns:
        Cleaned DataFrame with processed features
    """
    try:
        logger.info(f"Loading data from {file_path}...")
        # Load data
        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} rows")
        
        # Extract numeric ratings
        df['rating_numeric'] = df['Rating'].str.extract(r'Rated (\d) out of 5 stars').astype(float)
        
        # Drop rows with missing review text or rating
        df = df.dropna(subset=['Review Text', 'rating_numeric']).copy()
        logger.info(f"After dropping missing values: {len(df)} rows")
        df['rating_numeric'] = df['rating_numeric'].astype(int)
        
        # Extract reviewer experience
        df['reviewer_experience'] = df['Review Count'].str.extract(r'(\d+)').astype(float).fillna(1)
        
        # Combine title and review text
        df['full_text'] = df['Review Title'].fillna('') + ' ' + df['Review Text'].fillna('')
        df['full_text'] = df['full_text'].str.strip()
        
        # Create sentiment labels
        sentiment_map = {1: 'Very Bad', 2: 'Bad', 3: 'Neutral', 4: 'Good', 5: 'Very Good'}
        df['sentiment_label'] = df['rating_numeric'].map(sentiment_map)
        
        # Additional features
        df['exclamation_count'] = df['Review Text'].str.count('!')
        df['question_count'] = df['Review Text'].str.count(r'\?')
        df['uppercase_ratio'] = df['Review Text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        # Preprocess text
        logger.info("Preprocessing text...")
        from tqdm import tqdm
        tqdm.pandas()
        df['clean_text'] = df['full_text'].progress_apply(
            lambda x: preprocess_text(x, use_pos_tagging)
        )
        
        # Calculate word statistics
        df['clean_word_count'] = df['clean_text'].str.split().str.len()
        df['avg_word_length'] = df['clean_text'].apply(
            lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Remove empty texts and duplicates
        df = df[df['clean_text'].str.len() > 0].copy()
        df = df.drop_duplicates(subset=['clean_text'])
        logger.info(f"After removing duplicates: {len(df)} rows")
        
        # Select final columns
        final_cols = ['clean_text', 'rating_numeric', 'sentiment_label', 'reviewer_experience',
                      'clean_word_count', 'exclamation_count', 'question_count', 
                      'uppercase_ratio', 'avg_word_length']
        
        df_final = df[final_cols].copy()
        df_final.columns = ['text', 'rating', 'sentiment', 'reviewer_experience',
                            'word_count', 'exclamation_count', 'question_count', 
                            'uppercase_ratio', 'avg_word_length']
        
        logger.info(f"Preprocessing complete. Final shape: {df_final.shape}")
        return df_final
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading and cleaning data: {e}")
        raise


def map_to_3_classes(rating):
    """
    Map 5-class ratings to 3-class (Negative, Neutral, Positive).
    
    Args:
        rating: Numeric rating (1-5)
        
    Returns:
        String label: 'Negative', 'Neutral', or 'Positive'
    """
    if rating <= 2:
        return 'Negative'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Positive'