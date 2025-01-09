from keras.src.utils import pad_sequences
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

EMBEDDING_DIM = 500
EPOCHS = 17

df_train = pd.read_csv('../processed_train.csv')
tokenized_sentences = [text.split() for text in df_train['Cleaned_text']]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4)
word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=EPOCHS)
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}


def text_to_sequence(text, word_index):
    return [word_index[word] for word in text.split() if word in word_index]


sequences = [text_to_sequence(text, word_index) for text in df_train['Cleaned_text']]
padded_sequences = pad_sequences(sequences, maxlen=EMBEDDING_DIM, padding='post')

X_train = padded_sequences
y_train = df_train['Categorized_label']

df_test = pd.read_csv('../processed_test.csv')
tokenized_sentences_test = [text.split() for text in df_test['Cleaned_text']]
word2vec_model_test = Word2Vec(sentences=tokenized_sentences_test, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4)
word2vec_model_test.train(tokenized_sentences_test, total_examples=len(tokenized_sentences_test), epochs=EPOCHS)
word_index_test = {word: i + 1 for i, word in enumerate(word2vec_model_test.wv.index_to_key)}
sequences_test = [text_to_sequence(text, word_index_test) for text in df_test['Cleaned_text']]
padded_sequences_test = pad_sequences(sequences_test, maxlen=EMBEDDING_DIM, padding='post')
X_test = padded_sequences_test

def get_weight_matrix():
    vocab_size = len(word_index) + 1
    vector_size = word2vec_model.vector_size
    weight_matrix = np.zeros((vocab_size, vector_size))

    for word, i in word_index.items():
        if word in word2vec_model.wv:
            weight_matrix[i] = word2vec_model.wv[word]

    return weight_matrix


# Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


# Model architecture
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], trainable=False),
    Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(units=3, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Computing class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict = {int(k): v for k, v in class_weights_dict.items()}
y_train = np.array(y_train, dtype=int)


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    class_weight=class_weights_dict
)


# Generate predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)


def reverse_numeric_label(label):
    if label == 0:
        return 'fake'
    elif label == 1:
        return 'biased'
    else:
        return 'true'


df_test['Label'] = [reverse_numeric_label(label) for label in y_pred]
df_test = df_test.drop(columns=['Cleaned_text'])
df_test.to_csv('labelled_test.csv', index=False)
