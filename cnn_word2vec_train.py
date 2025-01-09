from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import seaborn as sns

EMBEDDING_DIM = 500
EPOCHS = 50

df = pd.read_csv('./processed_train.csv')
tokenized_sentences = [text.split() for text in df['Cleaned_text']]
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4)
word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=EPOCHS)
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}


def text_to_sequence(text, word_index):
    return [word_index[word] for word in text.split() if word in word_index]


sequences = [text_to_sequence(text, word_index) for text in df['Cleaned_text']]
padded_sequences = pad_sequences(sequences, maxlen=EMBEDDING_DIM, padding='post')  # Pad to max length

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences,
    df['Categorized_label'],
    test_size=0.2,
    random_state=42
)


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

# Early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

# Computing class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict = {int(k): v for k, v in class_weights_dict.items()}
print("class_weights: ", class_weights_dict)
y_train = np.array(y_train, dtype=int)


# Train the model
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

model.save('my_model.keras')

print("Class distribution:", np.bincount(y_train))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

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

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Fake', 'Biased', 'Real'],
            yticklabels=['Fake', 'Biased', 'Real'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('./confusion_matrix.png')

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Fake', 'Biased', 'Real']))

