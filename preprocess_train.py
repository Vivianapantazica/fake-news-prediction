import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.fr import French


def categorize_label(label):
    if label == 'fake':
        return 0
    elif label == 'biased':
        return 1
    else:
        return 2


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


df = pd.read_csv('../train.csv')
df['Categorized_label'] = df['Label'].apply(categorize_label)

# Check distribution of categories
print(df['Categorized_label'].value_counts())
df['Cleaned_text'] = df['Text'].apply(preprocess_french_text)
df = df.drop(columns=['Text', 'Label'])
df.to_csv('./processed_train.csv', index=False)