
# coding: utf-8

# In[1]:


import os


# In[2]:


os.listdir()


# In[3]:


pathEn = "/Users/kram/Downloads/botOrNot-en_es/en"


# First of all, we will create a procedure to be tested on a single file.
# 
# After this first step will be completed, we will extend this procedure to create a complete dataframe

# In[4]:


testfile = "/Users/kram/Downloads/botOrNot-en_es/en/1a5b808546838869bc39cebdbad951e3.xml"


# In[5]:


import pandas as pd
import xml.etree.ElementTree as ET
import io

def iter_docs(author):
    '''This function extracts the text and the language from the XML'''
    author_attr = author.attrib
    for doc in author.iter('document'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict

xml_data = open(testfile, "r") # Opening the text file
etree = ET.parse(xml_data) # Create an ElementTree object 
df = pd.DataFrame(list(iter_docs(etree.getroot()))) #Append the info to a pandas dataframe


# In[6]:


df.head(10)


# In[7]:


df['data'][0]


# In[8]:


# Getting ID to insert in the dataframe

filename = testfile.split("/")[-1].split(".")[0]


# In[9]:


filename


# We can now try to extend the procedure to the full directory.

# In[10]:


import time

# Creating empty dataframe
dataEn = pd.DataFrame()

# Monitoring time to load the files
start = time.time()

for root, dirs, files in os.walk(pathEn):
    for file in files:
        if file == 'truth.txt':
            continue
        else: 
            try:
                pathToFile = root + '/' + file # Creating path
                # print(pathToFile) # Just for debugging
                xml_data = open(pathToFile, "r", encoding="utf8") # Opening the text file
                etree = ET.parse(xml_data) # Create an ElementTree object
                data = list(iter_docs(etree.getroot())) # Create a list of dictionaries with the data
                filename = file.split(".")[0] # Get filename
                for dictionary in data: # Loop through the dictionary
                    dictionary['ID'] = filename # Append filename
                dataEn = dataEn.append(data)  # Append the list of dictionary to a pandas dataframe
                
            # If the file is not valid, skip it
            except ValueError as e:
                print(e)
                continue
            
end = time.time()
print("Total running time is", end - start)


# In[11]:


dataEn.head(10)


# In[12]:


dataEn['data'][0]


# In[13]:


dataEn.describe()


# Now that we have merged the IDs with the data, we can create another dataframe with the labels and then merge them using the ID as key

# In[14]:


pathToLabels = "/Users/kram/Downloads/botOrNot-en_es/en/truth.txt"


# In[15]:


target = pd.read_csv(pathToLabels, sep=":::")
target.columns=['ID', 'botOrHuman', 'sex'] 


# In[16]:


target.head(10)


# In[17]:


target.describe()


# We can now proceed with the concatenation of the dataframes for the English language

# In[18]:


mergedEnData = pd.merge(dataEn, target, on='ID')


# In[19]:


mergedEnData.head(10)


# In[20]:


mergedEnData.describe()


# # Deep Learning - CNN?

# In[21]:


'''Creo la litsa degli ID, delle classi e dei tweets pr ogni ID'''

listaIds =[]
listaClasses = []
matrixTweets = []

for index, x in mergedEnData.iterrows():
    id = x['ID']
    if id not in listaIds:
        newList = list()
        newList.append(x[1])
        matrixTweets.append(newList)
        listaIds.append(id)
        listaClasses.append(x[3])
    else:
        ls = matrixTweets[listaIds.index(id)]
        ls.append(x[1])
        matrixTweets[listaIds.index(id)] = ls
        
print(len(listaIds))


# In[41]:


'''Trasformo le entità, lascio le faccine, levo le stopword e se serve agli embeddings lemmatizzo'''


# In[42]:


import sys
get_ipython().system('{sys.executable} -m pip install ekphrasis')


# In[43]:


from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer

text_processor = TextPreProcessor (
    # terms that will be normalized
    normalize=[ 'email' , 'percent' , 'money' , 'phone' ,
                'time' , 'url' , 'date' , 'number' ] ,
    fix_html=True ,  # fix HTML tokens
    segmenter="twitter" ,
    corrector="twitter" ,
    unpack_hashtags=True ,  # perform word segmentation on hashtags
    unpack_contractions=True ,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True ,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[ emoticons ]
)


# In[44]:


'''Transform sentences to word embeddings'''


# In[45]:


import gensim


# In[46]:


google_300 = gensim.models.KeyedVectors.load_word2vec_format( "/Users/kram/Downloads/botOrNot-en_es/google_w2v_300.bin" , binary=True )


# In[47]:


'''Trasformo le frasi'''


# In[104]:


from nltk.tokenize import TweetTokenizer as TweetTokenizer
from nltk.corpus import stopwords
import random as rn
stop_words = set(stopwords.words('english'))

# matrixTweetsEmb = []
# for tweetsUser in matrixTweets:
#     embTweetsUser = []
#
#     for tweet in tweetsUser:
#         embTweetUser = np.zeros([50,300])
#         #Preprocesso
#         tokList = text_processor.pre_process_doc(tweet)
#         #Rimuovo le stopwords
#         tokList = [w for w in tokList if not w in stop_words]
#         #trovo l'embedding
#         numTok = 0;
#         for token in tokList[0:50]:
#             g_vec =[]
#             is_in_model = False
#             if token in google_300.vocab.keys ( ):
#                 is_in_model = True
#                 g_vec = google_300.word_vec(token)
#             elif token == "<number>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec( "number")
#             elif token == "<percent>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("percent")
#             elif token == "<money>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("money")
#             elif token == "<email>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("email")
#             elif token == "<phone>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("phone")
#             elif token == "<time>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("time")
#             elif token == "<date>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("date")
#             elif token == "<url>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec("url")
#             elif not is_in_model:
#                 max = len ( google_300.vocab.keys ( ) ) - 1
#                 index = rn.randint ( 0 , max )
#                 word = google_300.index2word[ index ]
#                 g_vec = google_300.word_vec( word )
#
#             embTweetUser[numTok] = np.array(g_vec)
#             numTok += 1
#         embTweetsUser.append(np.array(embTweetUser))
#
#
#
#     matrixTweetsEmb.append(np.array(embTweetsUser))
#
#
# # In[105]:
#
#
# '''Num Utenti x Num Tweets x Num MaxTokens x Dim Embedding'''
# import numpy as np
# matrixTweetsEmb = np.array(matrixTweetsEmb)
# print(matrixTweetsEmb.shape)


# In[22]:


get_ipython().system('{sys.executable} -m pip install joblib')
import joblib
#joblib.dump(matrixTweetsEmb,'matrixTweetsEmb_4177_100_50_300.dump')
matrixTweetsEmb = joblib.load('matrixTweetsEmb_4177_100_50_300.dump')


# In[23]:


from keras.layers import *
from keras.models import Sequential
model = Sequential()
model.add(Conv2D(200,(5,5), activation ='relu', input_shape=(100,50,300)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(100,(5,4), activation ='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(20,(3,3), activation ='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(400, activation="tanh"))
model.add(Dense(200, activation="tanh"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(2, activation="softmax"))
model.summary()


# In[24]:


get_ipython().system('{sys.executable} -m pip install category_encoders')
import category_encoders as ce
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
training_classes = le.fit_transform(listaClasses)
print(le.category_mapping)


# In[27]:


# model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )
#
# history = model.fit(matrixTweetsEmb,training_classes,64,15,
#                       validation_split= 0.15 ,
#                       verbose=1)
#
#
# # In[31]:
#
#
# import matplotlib.pyplot as plt
# print(history.history.keys())
# get_ipython().run_line_magic('matplotlib', 'qt')
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
#
# # In[29]:
#
#
# model.save('01.CNN_100x50x300D_google.h5')
#

# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrixTweetsEmb,training_classes, test_size=0.15, random_state=891)


# In[40]:


from  sklearn.metrics  import classification_report
from keras.callbacks import Callback
class MyCallBack(Callback):
    def __init__(self,verbose=0):

        super(Callback, self).__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_loss')
        if current < 0.014:
            self.model.stop_training = True
        
        predicted = self.model.predict_classes(X_test)
        print(classification_report(y_test,predicted))
        
        
callbacks_list = [
    MyCallBack(verbose=1)
]


# In[ ]:


model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )

from sklearn.model_selection import train_test_split, StratifiedKFold
folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=7654).split(matrixTweetsEmb,training_classes))

for j , (train_idx , val_idx) in enumerate ( folds ):
    print ( '\nFold ' , j )
    X_train = matrixTweetsEmb[ train_idx ]
    y_train = training_classes[ train_idx ]
    X_test = matrixTweetsEmb[ val_idx ]
    y_test = training_classes[ val_idx ]

    history = model.fit(X_train,y_train,64,15,
                          validation_data= (X_test,y_test) ,
                          callbacks=callbacks_list,
                          verbose=1)



model.save('02.CNN_100x50x300D_google.h5')