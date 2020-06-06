# Import the necessary modules
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import string
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import tensorflow as tf
import pickle
import warnings
import datetime
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings("ignore", category=DeprecationWarning)

#import data
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

fake.info()

#count subjects from fake dataset
sns.countplot(fake.subject)
plt.clf()
#explore title
fake.title.head()
#explore text
fake.text.head()

real.info()
#count subjects from real dataset
sns.countplot(real.subject) #only 2 subjects
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


#merge both datasets
df = pd.concat([real, fake])
#Drop columns
df.drop(["date","source","subject","title"], axis=1, inplace=True)

df.count()
# split into data to train and into "unseen" data
data_df=df.sample(frac=0.8,random_state=100) #random state is a seed value
unseen_df=df.drop(data_df.index).reset_index(drop=True)
data_df=data_df.reset_index(drop=True)
print(data_df.groupby('label').count())
print(unseen_df.groupby('label').count())

data_df.dtypes

def split_strings_n_words(df, n):
    new_df = pd.concat([pd.Series(row['label'], [' '.join(row['text'].split()[x:x + n]) for x in range(0, len(row['text'].split()), n)]) for _, row in df.iterrows()]).reset_index()
    # new data frame with split value columns
    new_df.rename(columns={"index":"text",0:"label"}, inplace=True)
    return new_df

data_df_500 = split_strings_n_words(df,500)
unseen_df_500=split_strings_n_words(unseen_df,500)

#initialise preprocessing parameters
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

# def preprocess_dataset(df):
#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(df.text)
#     sequence_text = tokenizer.texts_to_sequences(df.text)
#     padded_text = pad_sequences(sequence_text, maxlen=None)
#     targets = df.label
#     tf_idf = TfidfVectorizer(max_features=10000,stop_words={'english'})
#     tf_idf.fit(df.text)
#     tf_idf_vec = tf_idf.transform(df.text)
#
#     return padded_text, targets, tf_idf, tf_idf_vec

#basic preprocessing function for NN
def clean_data(dataframe):

    processed_corpus = []

    for i in range(len(dataframe)):

        text = str(dataframe['text'][i])
        #LOWERCASE TEXT
        text = text.lower()
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', str(text))
        # nfkd_form = unicodedata.normalize('NFKD', text)
        # text = nfkd_form.encode('ascii', 'ignore').decode()
        #REMOVE TAGS
        text = BeautifulSoup(str(text)).get_text()
        processed_corpus.append(str(text))
    return processed_corpus


def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

cleaned_documents = clean_data(data_df_500)

update_df(data_df_500, cleaned_documents)

len(df)
clean_docs_all = clean_data(df)
update_df(df, clean_docs_all)

#new data to predict on including labels
cleaned_documents_new=clean_data(unseen_df_500)
update_df(unseen_df_500,cleaned_documents_new)
unseen_df_500.insert(2,'predicted',value=None)

len(unseen_df)
clean_docs_all_new = clean_data(unseen_df)
update_df(unseen_df, clean_docs_all_new)
unseen_df

#EMBEDDING size tune

def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


tokenized_corpus = tokenize_corpus(data_df_500.text)
vocabulary = {word for doc in tokenized_corpus for word in doc}
print("corpora vocab length:{}".format(len(vocabulary)))


oov_tok = '<OOV>'
trunc_type = 'post'
padding_type = 'post'
embedding_dim=100
#set random seeds to obtain more or less reproducible results
random_seed=42
tf.random.set_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)


def LSTM_model(df,new_df,MAX_LEN,MAX_NB_WORDS,epochs,batch_size):

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True) # The maximum number of words to be used. (most frequent) or could use whole vocab size
    tokenizer.fit_on_texts(df.text.values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # Change text for numerical ids and pad
    X = tokenizer.texts_to_sequences(df.text)
    X = pad_sequences(X, maxlen=MAX_LEN) #max length of texts, used for padding
    print('Shape of data tensor:', X.shape)
    Y = df.label


    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.01, random_state=random_seed, stratify=Y) #test_size held small as holdout used instead

    # # Change text for numerical ids and pad
    # X_test = tokenizer.texts_to_sequences(test_df.text)
    # X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    # print(X_train.shape, y_train.shape)
    # print(X_dev.shape, y_dev.shape)

    # kfold_splits=10
    # # # Instantiate the cross validator
    # skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True,random_state=7)
    #
    # # # Loop through the indices the split() method returns
    #
    # for index, (train_indices, val_indices) in enumerate(skf.split(X, Y.values.argmax(1))):
    #     print
    #     "Training on fold " + str(index + 1) + "/10..."
    #     # Generate batches from indices
    #     X_train, X_dev = X[train_indices], X[val_indices]
    #     y_train, y_dev = Y[train_indices], Y[val_indices]

    # Clear model, and create it
    #baseline pre-LSTM
    model = Sequential()
    # # Add an Embedding layer expecting input , and output embedding dimension of size 100 we set at the top
    model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    #baseline LSTM
    # model.add(LSTM(50))

    #100 layer LSTM ( proved to be too computationally expensive and almost double required training time for one epoch)
    # model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    #v2
    model.add(LSTM(50))

    # #v3
    # model.add(Embedding(MAX_NB_WORDS,embedding_dim,input_length=X.shape[1]))
    # model.add(tf.keras.layers.Bidirectional(LSTM(embedding_dim)))
    # # use ReLU in place of tanh function since they are very good alternatives of each other.
    # model.add(Dense(embedding_dim, activation='relu'))

    #o baseline utput layer
    # Add a Dense layer with 1 unit and sigmod activation since we only have a binary output we use binary cross-entropy for our loss function
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_dev,y_dev),callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.05)])

    # evaluate the model
    train_acc = model.evaluate(X_train, y_train, verbose=0)

    test_acc = model.evaluate(X_dev, y_dev, verbose=0)
    print('Train set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(train_acc[0],train_acc[1]))
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(test_acc[0],test_acc[1]))

    # plot training history
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.title("{} sample".format(MAX_LEN))
    plt.savefig('lstm_model_{}_{}.png'.format(MAX_LEN, datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S")))
    plt.clf()
    # save the model to disk
    filename = 'lstm_model_{}_{}.sav'.format(MAX_LEN, datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
    pickle.dump(model, open(filename, 'wb'))

    # Change text for numerical ids and pad
    X_new = tokenizer.texts_to_sequences(new_df.text)
    X_new = pad_sequences(X_new, maxlen=MAX_LEN)

    # Use the model to predict on new data
    predicted = model.predict(X_new)

    # # Choose the class with higher probability
    y_pred = (predicted > 0.5)
    new_df['predicted'] = y_pred

    # Create the performance report
    print(classification_report(new_df['label'], new_df['predicted']))

    return predicted, history,model

#with 1000 sample dataset
# pred=LSTM_model(balanced_train_1000, balanced_test_1000, file_data_new,1000,30000,10,32)

#with 500 sample dataset, parameters for the results presented in the report
pred_slow_1,history_slow_1,model_slow_1 = LSTM_model(data_df_500, unseen_df_500,500,100000,1,100)
pred_slow_BS,history_slow_BS,model_slow_BS = LSTM_model(data_df_500, unseen_df_500,500,100000,10,100)
pred_slow_100LS,history_slow_100LS,model_slow_100LS = LSTM_model(data_df_500, unseen_df_500,500,100000,10,100)
pred_slow_25LS,history_slow_25LS,model_slow_25LS = LSTM_model(data_df_500, unseen_df_500,500,100000,10,100)

pred_slow_1_fl,history_slow_1_fl,model_slow_1_fl = LSTM_model(df, unseen_df, 500, 100000,1,100)

pred_slow_5,history_slow_5 = LSTM_model(data_df_500, unseen_df_500,500,100000,5,32)

