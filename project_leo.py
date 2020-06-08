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
#explore title
fake.title.head()
#explore text
fake.text.head()

real.info()
#count subjects from real dataset
sns.countplot(real.subject) #only 2 subjects

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


data_df_500 = split_strings_n_words(data_df,1000)
unseen_df_500=split_strings_n_words(unseen_df,1000)

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
        text = dataframe['text'][i]
        #LOWERCASE TEXT
        text = text.lower()
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z-ÁÀÂÃâáàãçÉÈÊéèêúùÚÙÕÓÒÔôõóòÍÌíìçÇ]", ' ', text)
        # nfkd_form = unicodedata.normalize('NFKD', text)
        # text = nfkd_form.encode('ascii', 'ignore').decode()
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        processed_corpus.append(text)
    return processed_corpus


def update_df(dataframe, cleaned_documents):
    dataframe['text'] = cleaned_documents

cleaned_documents = clean_data(data_df_500)

update_df(data_df_500, cleaned_documents)

#new data to predict on including labels
cleaned_documents_new=clean_data(unseen_df_500)
update_df(unseen_df_500,cleaned_documents_new)
unseen_df_500.insert(2,'predicted',value=None)
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

def processing(df,new_df,MAX_LEN, MAX_NB_WORDS):
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
    return X_train, X_dev,y_train, y_dev,MAX_NB_WORDS,MAX_LEN, X, X_new

def run_model(MAX_LEN,epochs,batch_size,model, X_train, X_dev, y_train, y_dev,X_new):
    print("epochs: ",epochs, "batch size: ",batch_size)

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


    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(X_dev,y_dev),callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.05)])

    # accuracy_history = history.history['acc']
    # val_accuracy_history = history.history['val_acc']
    # print( "Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]))

    # save the model to disk
    # filename = 'lstm_model_{}_{}.pkl'.format(MAX_LEN,datetime.datetime.today().strftime("%d_%m_%Y_%H_%M_%S"))
    # pickle.dump(model, open(filename, 'wb'))
    # # # load the model from disk
    # model_name='lstm_model_1000_06_04_2020_11_50_20.pkl'
    # loaded_model = pickle.load(open(model_name, 'rb'))
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
    plt.show()




    # Use the model to predict on new data
    predicted = model.predict(X_new)

    # # Choose the class with higher probability
    y_pred = (predicted > 0.5)
    unseen_df_500['predicted'] = y_pred

    # Create the performance report
    print(classification_report(unseen_df_500['label'], unseen_df_500['predicted']))

    return predicted, history

X_train, X_dev,y_train, y_dev,MAX_NB_WORDS,MAX_LEN,X, X_new= processing(data_df_500, unseen_df_500,500,100000)
# Clear model, and create it
#v1
modelLSTM = Sequential()
modelLSTM.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=100, input_length=X.shape[1]))
#v2
modelLSTM.add(SpatialDropout1D(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
modelLSTM.add(LSTM(50))
# #v3
# # Add an Embedding layer expecting input , and output embedding dimension of size 100 we set at the top
# model.add(tf.keras.layers.Embedding(MAX_NB_WORDS,embedding_dim,input_length=X.shape[1]))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
# # use ReLU in place of tanh function since they are very good alternatives of each other.
# model.add(tf.keras.layers.Dense(embedding_dim, activation='relu'))

#output layer
# Add a Dense layer with 6 units and softmax activation.When we have multiple outputs, softmax convert outputs layers into a probability distribution.

modelLSTM.add(Dense(1, activation='sigmoid'))
modelLSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


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
    modelconv.add(Conv1D(conv,5,activation='relu'))
    modelconv.add(MaxPool1D(3))
    modelconv.add(Conv1D(conv,5,activation='relu'))
    modelconv.add(MaxPool1D(3))
    modelconv.add(Conv1D(conv,5,activation='relu'))
    modelconv.add(GlobalMaxPooling1D())
    modelconv.add(Dense(1, activation='sigmoid'))
    modelconv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelconv
#with 1000 sample dataset
# pred=LSTM_model(balanced_train_1000, balanced_test_1000, file_data_new,1000,30000,10,32)
#with 500 sample dataset, parameters for the results presented in the report
print("model: SimpleRNN")
print("size: 50")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_GRU(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,10,32,mod_GRU(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,1,100,mod_GRU(50), X_train, X_dev, y_train, y_dev,X_new)
print("size: 25")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_GRU(25), X_train, X_dev, y_train, y_dev,X_new)
print("size: 100")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_GRU(100), X_train, X_dev, y_train, y_dev,X_new)

print("model: SimpleRNN")
print("size: 50")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_simple(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,10,32,mod_simple(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,1,100,mod_simple(50), X_train, X_dev, y_train, y_dev,X_new)
print("size: 25")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_simple(25), X_train, X_dev, y_train, y_dev,X_new)
print("size: 100")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_simple(100), X_train, X_dev, y_train, y_dev,X_new)


print("model: Conv1D")
print("size: 50")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_conv(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,10,32,mod_conv(50), X_train, X_dev, y_train, y_dev,X_new)
pred_slow,history_slow=run_model(MAX_LEN,1,100,mod_conv(50), X_train, X_dev, y_train, y_dev,X_new)
print("size: 25")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_conv(25), X_train, X_dev, y_train, y_dev,X_new)
print("size: 100")
pred_slow,history_slow=run_model(MAX_LEN,10,100,mod_conv(100), X_train, X_dev, y_train, y_dev,X_new)

#1536s 26ms/step - loss: 0.1685 - accuracy: 0.9366 - val_loss: 0.3577 - val_accuracy: 0.8265
# Train set
#   Loss: 0.329
#   Accuracy: 0.854
# Test set
#   Loss: 0.358
#   Accuracy: 0.827

# X_new prediction performance
#               precision    recall  f1-score   support
#            0       0.87      0.92      0.90      1811
#            1       0.85      0.77      0.81      1067
#     accuracy                           0.87      2878
#    macro avg       0.86      0.85      0.85      2878
# weighted avg       0.87      0.87      0.86      2878
