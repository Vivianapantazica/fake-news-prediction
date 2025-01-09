import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.fr import French


def preprocess_french_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-alphanumeric characters, preserve accents
    text = re.sub(r'[^\w\sàâçéèêëîïôûùüÿñæœ]', '', text)
    # Tokenize and handle contractions (use spaCy French tokenizer)
    nlp = French()
    doc = nlp(text)
    words = [token.text for token in doc if token.is_alpha]
    # Remove stopwords
    stop_words = set(stopwords.words('french'))
    stop_words.update(['plus'])
    words = [word for word in words if word not in stop_words]
    # Apply stemming
    stemmer = SnowballStemmer('french')
    words = [stemmer.stem(word) for word in words if len(word) > 3]
    # Join words back into a single string
    return ' '.join(words)


df = pd.read_csv('../test.csv')
df['Cleaned_text'] = df['Text'].apply(preprocess_french_text)
df.to_csv('./processed_test.csv', index=False)