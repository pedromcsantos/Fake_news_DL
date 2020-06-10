# Import the necessary modules
from bs4 import BeautifulSoup
import string
import re
import random
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
warnings.filterwarnings("ignore", category=DeprecationWarning)

#define functions
# basic preprocessing function for NN
def clean_data(dataframe):
    processed_corpus = []
    for i in range(len(dataframe)):
        text = str(dataframe['text'].loc[i])
        # LOWERCASE TEXT
        text = text.lower()
        # REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', str(text))
        # REMOVE TAGS
        text = BeautifulSoup(str(text)).get_text()
        processed_corpus.append(str(text))

    return processed_corpus


def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents


def split_strings_n_words(df, n):
    new_df = pd.concat([pd.Series(row['label'], [' '.join(row['text'].split()[x:x + n]) for x in
                                                 range(0, len(row['text'].split()), n)]) for _, row in
                        df.iterrows()]).reset_index()
    # new data frame with split value columns
    new_df.rename(columns={"index": "text", 0: "label"}, inplace=True)
    return new_df


# EMBEDDING size tune
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def processing(df, new_df, MAX_LEN, MAX_NB_WORDS):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          lower=True)  # The maximum number of words to be used. (most frequent) or could use whole vocab size
    tokenizer.fit_on_texts(df.text.values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Change text for numerical ids and pad
    X = tokenizer.texts_to_sequences(df.text)
    X = pad_sequences(X, maxlen=MAX_LEN)  # max length of texts, used for padding
    print('Shape of data tensor:', X.shape)
    Y = df.label

    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.01, random_state=random_seed,
                                                      stratify=Y)  # test_size held small as holdout used instead
    # Change text for numerical ids and pad
    X_new = tokenizer.texts_to_sequences(new_df.text)
    X_new = pad_sequences(X_new, maxlen=MAX_LEN)
    return X_train, X_dev, y_train, y_dev, MAX_NB_WORDS, MAX_LEN, X, X_new


def LSTM_model(df, new_df, MAX_LEN, MAX_NB_WORDS, epochs, batch_size):
    # with 500 sample dataset, parameters for the results presented in the report
    X_train, X_dev, y_train, y_dev, MAX_NB_WORDS, MAX_LEN, X, X_new = processing(df, new_df, MAX_LEN, MAX_NB_WORDS)

    # Clear model, and create it
    # baseline pre-LSTM
    model = Sequential()
    # # Add an Embedding layer expecting input , and output embedding dimension of size 100 we set at the top
    model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))

    # baseline LSTM (50 neurons)
    model.add(LSTM(50))

    # 100 neuron LSTM ( proved to be too computationally expensive and almost double required training time for one epoch)
    # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    # 25 neuron
    # model.add(LSTM(25))

    # to baseline output layer
    # Add a Dense layer with 1 unit and sigmod activation since we only have a binary output we use binary cross-entropy for our loss function
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev),
                        callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.05)])

    # evaluate the model
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_dev, y_dev, verbose=0)
    print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc[0], train_acc[1]))
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc[0], test_acc[1]))

    # plot training history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.title("{} sample".format(MAX_LEN))
    plt.savefig('lstm_model_{}_{}.png'.format(MAX_LEN, datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S")))
    # save the model to disk
    filename = 'lstm_model_{}_{}.sav'.format(MAX_LEN, datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
    pickle.dump(model, open(filename, 'wb'))

    # Use the model to predict on new data
    predicted = model.predict(X_new)

    # # Choose the class with higher probability
    y_pred = (predicted > 0.5)
    new_df['predicted'] = y_pred

    # Create the performance report
    print(classification_report(new_df['label'], new_df['predicted']))
    return predicted, history, model

def run_model(MAX_LEN, epochs, batch_size, model, X_train, X_dev, y_train, y_dev, X_new):
    print("epochs: ", epochs, "batch size: ", batch_size)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_dev, y_dev),
                        callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.05)])

    train_acc = model.evaluate(X_train, y_train, verbose=0)

    test_acc = model.evaluate(X_dev, y_dev, verbose=0)
    print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc[0], train_acc[1]))
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc[0], test_acc[1]))

    # plot training history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.title("{} sample".format(MAX_LEN))
    plt.show()

    # Use the model to predict on new data
    predicted = model.predict(X_new)

    # # Choose the class with higher probability
    y_pred = (predicted > 0.5)
    unseen_df_500['predicted'] = y_pred

    # Create the performance report
    print(classification_report(unseen_df_500['label'], unseen_df_500['predicted']))

    return predicted, history


def mod_simple(size):
    from keras.layers import SimpleRNN
    model_simple = Sequential()
    model_simple.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    model_simple.add(SpatialDropout1D(0.2))
    model_simple.add(SimpleRNN(size))
    model_simple.add(Dense(1, activation='sigmoid'))
    model_simple.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_simple


def mod_GRU(size):
    from keras.layers import GRU
    modelGRU = Sequential()
    modelGRU.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    modelGRU.add(SpatialDropout1D(0.2))
    modelGRU.add(GRU(size))
    modelGRU.add(Dense(1, activation='sigmoid'))
    modelGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelGRU


def mod_conv(conv):
    from keras.layers import Conv1D, MaxPool1D, GlobalMaxPooling1D
    modelconv = Sequential()
    modelconv.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    modelconv.add(Conv1D(conv, 5, activation='relu'))
    modelconv.add(MaxPool1D(3))
    modelconv.add(Conv1D(conv, 5, activation='relu'))
    modelconv.add(MaxPool1D(3))
    modelconv.add(Conv1D(conv, 5, activation='relu'))
    modelconv.add(GlobalMaxPooling1D())
    modelconv.add(Dense(1, activation='sigmoid'))
    modelconv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelconv


def mod_dense(size):
    from keras.layers import Dense, Dropout, Flatten
    modeldense = Sequential()
    modeldense.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    modeldense.add(Flatten())
    modeldense.add(Dense(size, activation='relu'))
    modeldense.add(Dropout(0.2))
    modeldense.add(Dense(size, activation='relu'))
    modeldense.add(Dropout(0.2))
    modeldense.add(Dense(size, activation='relu'))
    modeldense.add(Dropout(0.2))
    modeldense.add(Dense(1, activation='sigmoid'))
    modeldense.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modeldense

# import data
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

fake.info()

# count subjects from fake dataset
sns.countplot(fake.subject)
plt.clf()
# explore title
fake.title.head()
# explore text
fake.text.head()

real.info()
# count subjects from real dataset
sns.countplot(real.subject)  # only 2 subjects
plt.clf()

source = []
new_text = []
for row in real.text:
    try:
        record = row.split(" -", maxsplit=1)
        source.append(record[0])
        new_text.append(record[1])
    except:
        new_text.append(record[0])

real["source"] = source
real["text"] = new_text

real["label"] = 1
fake["label"] = 0

# merge both datasets
df = pd.concat([real, fake])
# Drop columns
df.drop(["date", "source", "subject", "title"], axis=1, inplace=True)

# split into data to train and into "unseen" data
data_df = df.sample(frac=0.8, random_state=100)  # random state is a seed value
unseen_df = df.drop(data_df.index).reset_index(drop=True)
data_df = data_df.reset_index(drop=True)
print(data_df.groupby('label').count())
print(unseen_df.groupby('label').count())

data_df_1000 = split_strings_n_words(data_df, 1000)
unseen_df_1000 = split_strings_n_words(unseen_df, 1000)

data_df_500 = split_strings_n_words(data_df, 500)
unseen_df_500 = split_strings_n_words(unseen_df, 500)

exclude = set(string.punctuation)

# preprocess train data
cleaned_documents = clean_data(data_df_500)
update_df(data_df_500, cleaned_documents)

cleaned_documents_1000 = clean_data(data_df_1000)
update_df(data_df_1000, cleaned_documents_1000)

clean_docs_all = clean_data(data_df)
update_df(data_df, clean_docs_all)

# cleaning new data to predict on excluding labels
cleaned_documents_new = clean_data(unseen_df_500)
update_df(unseen_df_500, cleaned_documents_new)
unseen_df_500.insert(2, 'predicted', value=None)

cleaned_documents_new_1000 = clean_data(unseen_df_1000)
update_df(unseen_df_1000, cleaned_documents_new_1000)
unseen_df_1000.insert(2, 'predicted', value=None)

clean_docs_all_new = clean_data(unseen_df)
update_df(unseen_df, clean_docs_all_new)
unseen_df.insert(2, 'predicted', value=None)

#tokenising the corpus to understand vocab length to set a sensible MAX_NB_WORDS

tokenized_corpus = tokenize_corpus(data_df_500.text)
vocabulary = {word for doc in tokenized_corpus for word in doc}
print("corpora vocab length:{}".format(len(vocabulary)))

max_full = max(max(unseen_df['text'].str.split().str.len()), max(data_df['text'].str.split().str.len()))

embedding_dim = 100
# set random seeds to obtain more or less reproducible results
random_seed = 42
tf.random.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

# all LSTM model runs with 500-word-split
pred_slow_1, history_slow_1, model_slow_1 = LSTM_model(data_df_500, unseen_df_500, 500, 100000, 1, 100)
pred_slow_BS, history_slow_BS, model_slow_BS = LSTM_model(data_df_500, unseen_df_500, 500, 100000, 10, 100)
pred_slow_100LS, history_slow_100LS, model_slow_100LS = LSTM_model(data_df_500, unseen_df_500, 500, 100000, 10, 100)
pred_slow_25LS, history_slow_25LS, model_slow_25LS = LSTM_model(data_df_500, unseen_df_500, 500, 100000, 10, 100)

# all LSTM model runs with 1000-word-split
pred_slow_1, history_slow_1, model_slow_1 = LSTM_model(data_df_1000, unseen_df_1000, 1000, 100000, 1, 100)
pred_slow_BS, history_slow_BS, model_slow_BS = LSTM_model(data_df_1000, unseen_df_1000, 1000, 100000, 10, 100)
pred_slow_100LS, history_slow_100LS, model_slow_100LS = LSTM_model(data_df_1000, unseen_df_1000, 1000, 100000, 10, 100)
pred_slow_25LS, history_slow_25LS, model_slow_25LS = LSTM_model(data_df_1000, unseen_df_1000, 1000, 100000, 10, 100)

# predicted 6 hours to run for full dataset (didn't run for report)
pred_slow_1_fl, history_slow_1_fl, model_slow_1_fl = LSTM_model(df, unseen_df, max_full, 100000, 1, 100)


# with 500 sample dataset, parameters for the results presented in the report
X_train, X_dev, y_train, y_dev, MAX_NB_WORDS, MAX_LEN, X, X_new = processing(data_df_500, unseen_df_500, 500, 100000)

print("model: SimpleRNN")
print("size: 50")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_GRU(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 10, 32, mod_GRU(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 1, 100, mod_GRU(50), X_train, X_dev, y_train, y_dev, X_new)
print("size: 25")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_GRU(25), X_train, X_dev, y_train, y_dev, X_new)
print("size: 100")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_GRU(100), X_train, X_dev, y_train, y_dev, X_new)

print("model: SimpleRNN")
print("size: 50")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_simple(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 10, 32, mod_simple(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 1, 100, mod_simple(50), X_train, X_dev, y_train, y_dev, X_new)
print("size: 25")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_simple(25), X_train, X_dev, y_train, y_dev, X_new)
print("size: 100")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_simple(100), X_train, X_dev, y_train, y_dev, X_new)

print("model: Conv1D")
print("size: 50")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_conv(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 10, 32, mod_conv(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 1, 100, mod_conv(50), X_train, X_dev, y_train, y_dev, X_new)
print("size: 25")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_conv(25), X_train, X_dev, y_train, y_dev, X_new)
print("size: 100")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_conv(100), X_train, X_dev, y_train, y_dev, X_new)

print("model: Dense")
print("size: 50")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 10, 32, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 1, 100, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
print("size: 25")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(25), X_train, X_dev, y_train, y_dev, X_new)
print("size: 100")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(100), X_train, X_dev, y_train, y_dev, X_new)

# with 500 sample dataset, parameters for the results presented in the report
X_train, X_dev, y_train, y_dev, MAX_NB_WORDS, MAX_LEN, X, X_new = processing(data_df_1000, unseen_df_1000, 1000, 100000)

print("model: Dense")
print("size: 50")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 10, 32, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
pred_slow, history_slow = run_model(MAX_LEN, 1, 100, mod_dense(50), X_train, X_dev, y_train, y_dev, X_new)
print("size: 25")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(25), X_train, X_dev, y_train, y_dev, X_new)
print("size: 100")
pred_slow, history_slow = run_model(MAX_LEN, 10, 100, mod_dense(100), X_train, X_dev, y_train, y_dev, X_new)
