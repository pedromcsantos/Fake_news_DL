#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


import warnings
warnings.filterwarnings('ignore')


# In[3]:


real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")


# ## Explore Fake News Dataset

# In[4]:


fake.info()


# In[5]:


sns.countplot(fake.subject)


# In[6]:


fake.title.head()


# In[7]:


fake.text.head()


# ## Explore Real News Dataset

# In[8]:


real.info()


# In[9]:


sns.countplot(real.subject) #less subjects


# In[10]:


real.title.head()


# In[11]:


real.text.head() 
#real news seem to always start with the publisher/source. since fake news dont have it, we should remove them
#source can also be twitter


# In[12]:


source = []
new_text = []
for row in real.text:
    try:
        record = row.split(" -", maxsplit=1)
        source.append(record[0])
        new_text.append(record[1])
    except:
         new_text.append(record[0])


# In[13]:


real["source"] = source
real["text"] = new_text #replace the previous text with the new one without source


# In[14]:


real


# ## Pre-processing

# In[15]:


#Create label for real/fake
real["label"] = 1
fake["label"] = 0


# In[16]:


#merge both datasets
df = pd.concat([real, fake])


# In[17]:


#2 approaches: with Text only or with Text + Title

#if text + title
text_title = True
if text_title == True:
    df["text"] = df["title"] + df["text"]
       


# In[18]:


#Drop columns
df.drop(["date","source","subject","title"], axis=1, inplace=True)


# In[19]:


#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_dataset(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)
    sequence_text = tokenizer.texts_to_sequences(df.text)
    padded_text = pad_sequences(sequence_text, maxlen=None)
    targets = df.label
    tf_idf = TfidfVectorizer(max_features=10000,stop_words={'english'})
    tf_idf_fit = tf_idf.fit(sequence_text)
    tf_idf_vec = tf_idf.transform(sequence_text)

    return padded_text, targets, tf_idf, tf_idf_vec
"""
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    filtered_texts = []
    for i in sequence_text:
        filtered_sentence = [w for w in i if not w in stop_words]
        filtered_texts.append(filtered_sentence)
"""


# In[20]:


def model(embedding_dim=128):
    model = Sequential()
    model.add(Embedding(10000, embedding_dim))
    model.add(LSTM(embedding_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[21]:


def model_history(model, X_train,y_train, epochs, batch_size):

    history = model.fit(X_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=0.2)
    return history


# In[22]:


model().summary()


# In[ ]:


padded_text, targets, tf_idf, tf_idf_vec = preprocess_dataset(df)

def splits(text, target):
    X_train_val, X_test, y_train_val, y_test = train_test_split(text,target,stratify=target,test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(text,target,stratify=target,test_size=0.3)
    return X_train,X_val,X_test,y_train,y_val,y_test

X_train,X_val,X_test,y_train,y_val,y_test = splits(padded_text,targets)
X_traintf,X_valtf,X_testtf,y_traintf,y_valtf,y_testtf = splits(tf_idf_vec)


# In[ ]:


model = model()


# In[ ]:


history = model_history(model, X_train, y_train, 1, 1000)


# In[ ]:




