import os
import pandas as pd
import xml.etree.ElementTree as ET
import io

pathEn = "/Users/kram/Downloads/botOrNot-en_es/es"

def iter_docs(author):
    '''This function extracts the text and the language from the XML'''
    author_attr = author.attrib
    for doc in author.iter('document'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict


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


pathToLabels = "/Users/kram/Downloads/botOrNot-en_es/es/truth.txt"

target = pd.read_csv(pathToLabels, sep=":::")
target.columns=['ID', 'botOrHuman', 'sex'] 

mergedEnData = pd.merge(dataEn, target, on='ID')

mergedEnData = mergedEnData[mergedEnData.botOrHuman == "human"]


# In[ ]:


pathEn = '/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es'
import time

# Creating empty dataframe
dataEn2 = pd.DataFrame()

# Monitoring time to load the files
start = time.time()

for root, dirs, files in os.walk(pathEn):
    for file in files:
        if file == 'truth.csv':
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
                dataEn2 = dataEn2.append(data)  # Append the list of dictionary to a pandas dataframe
                
            # If the file is not valid, skip it
            except ValueError as e:
                print(e)
                continue
            
end = time.time()
print("Total running time is", end - start)


# In[ ]:
pathToLabels = "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/truth.csv"


# In[ ]:


target = pd.read_csv(pathToLabels, sep=":::", header = None)
target.columns=['ID', 'botOrHuman', 'sex'] 


# In[ ]:
mergedEnData2 = pd.merge(dataEn2, target, on='ID')


# In[ ]:


#Creo le matrici


# In[12]:


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
        listaClasses.append(x[4])
    else:
        ls = matrixTweets[listaIds.index(id)]
        ls.append(x[1])
        matrixTweets[listaIds.index(id)] = ls
        
print(len(listaIds))

mergedEnData = None
# In[ ]:


listaIds2 =[]
listaClasses2 = []
matrixTweets2 = []

for index, x in mergedEnData2.iterrows():
    id = x['ID']
    if id not in listaIds2:
        newList = list()
        newList.append(x[1])
        matrixTweets2.append(newList)
        listaIds2.append(id)
        listaClasses2.append(x[4])
    else:
        ls = matrixTweets2[listaIds2.index(id)]
        ls.append(x[1])
        matrixTweets2[listaIds2.index(id)] = ls
        
print(len(listaIds2))
print(len(matrixTweets2))

mergedEnData2 = None
# In[13]:


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
    unpack_contractions=False ,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False ,  # spell correction for elongated words
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[ emoticons ]
)


# In[14]:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrixTweets, listaClasses, test_size = 0.20, random_state=999)


# In[ ]:


import numpy as np
X_train = np.concatenate([X_train,matrixTweets2])
y_train = np.concatenate([y_train,listaClasses2])

matrixTweets2 = None
import gc
gc.collect()
# In[ ]:

# In[15]:

import gensim
google_300 = gensim.models.KeyedVectors.load_word2vec_format("/Volumes/MacPassport/PycharmProjects/botOrNot/cc.es.300.vec")


# In[18]:


from nltk.tokenize import TweetTokenizer as TweetTokenizer
from nltk.corpus import stopwords
import random as rn
import numpy as np
stop_words = set(stopwords.words('spanish'))
i = 0
matrixTweetsEmb_train = np.zeros ( (len(X_train) ,100, 50, 300) )
for tweetsUser in X_train:
    embTweetsUser = []
    if(i % 100) == 0:
         print(i)
    for tweet in tweetsUser:
        embTweetUser = np.zeros([50,300])
        #Preprocesso
        tokList = text_processor.pre_process_doc(tweet)
        #Rimuovo le stopwords
        tokList = [w for w in tokList if not w in stop_words]
        #trovo l'embedding
        numTok = 0;
        for token in tokList[0:50]:
            g_vec =[]
            is_in_model = False
            if token in google_300.vocab.keys ( ):
                is_in_model = True
                g_vec = google_300.word_vec(token)
            elif token == "<number>":
                is_in_model = True
                g_vec = google_300.word_vec( "número")
            elif token == "<percent>":
                is_in_model = True
                g_vec = google_300.word_vec("porcentaje")
            elif token == "<money>":
                is_in_model = True
                g_vec = google_300.word_vec("dinero")
            elif token == "<email>":
                is_in_model = True
                g_vec = google_300.word_vec("email")
            elif token == "<phone>":
                is_in_model = True
                g_vec = google_300.word_vec("teléfono")
            elif token == "<time>":
                is_in_model = True
                g_vec = google_300.word_vec("hora")
            elif token == "<date>":
                is_in_model = True
                g_vec = google_300.word_vec("fecha")
            elif token == "<url>":
                is_in_model = True
                g_vec = google_300.word_vec("link")
            elif not is_in_model:
                max = len ( google_300.vocab.keys ( ) ) - 1
                index = rn.randint ( 0 , max )
                word = google_300.index2word[ index ]
                g_vec = google_300.word_vec( word )
            embTweetUser[ numTok ] = np.array ( g_vec )
            numTok += 1
        embTweetsUser.append ( np.array ( embTweetUser ) )
        embTweetUser = None
    matrixTweetsEmb_train[ i ] = np.array ( embTweetsUser )
    i += 1
    embTweetsUser = None
    gc.collect ( )


# In[19]:


i = 0
matrixTweetsEmb_test = np.zeros ( (len(X_test) ,100, 50, 300) )
for tweetsUser in X_test:
    embTweetsUser = []
    if(i % 100) == 0:
         print(i)
    for tweet in tweetsUser:
        embTweetUser = np.zeros([50,300])
        #Preprocesso
        tokList = text_processor.pre_process_doc(tweet)
        #Rimuovo le stopwords
        tokList = [w for w in tokList if not w in stop_words]
        #trovo l'embedding
        numTok = 0;
        for token in tokList[0:50]:
            g_vec =[]
            is_in_model = False
            if token in google_300.vocab.keys ( ):
                is_in_model = True
                g_vec = google_300.word_vec(token)
            elif token == "<number>":
                is_in_model = True
                g_vec = google_300.word_vec( "número")
            elif token == "<percent>":
                is_in_model = True
                g_vec = google_300.word_vec("porcentaje")
            elif token == "<money>":
                is_in_model = True
                g_vec = google_300.word_vec("dinero")
            elif token == "<email>":
                is_in_model = True
                g_vec = google_300.word_vec("email")
            elif token == "<phone>":
                is_in_model = True
                g_vec = google_300.word_vec("teléfono")
            elif token == "<time>":
                is_in_model = True
                g_vec = google_300.word_vec("hora")
            elif token == "<date>":
                is_in_model = True
                g_vec = google_300.word_vec("fecha")
            elif token == "<url>":
                is_in_model = True
                g_vec = google_300.word_vec("link")
            elif not is_in_model:
                max = len ( google_300.vocab.keys ( ) ) - 1
                index = rn.randint ( 0 , max )
                word = google_300.index2word[ index ]
                g_vec = google_300.word_vec( word )

            embTweetUser[ numTok ] = np.array ( g_vec )
            numTok += 1
        embTweetsUser.append ( np.array ( embTweetUser ) )
        embTweetUser = None
    matrixTweetsEmb_test[ i ] = np.array ( embTweetsUser )
    i += 1
    embTweetsUser = None
    gc.collect ( )


# In[20]:




# In[21]:


import category_encoders as ce
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
training_classes = le.fit_transform(y_train)
test_classes = le.transform(y_test)
print(le.category_mapping)


# In[22]:


from  sklearn.metrics  import classification_report
from keras.callbacks import Callback
class MyCallBack(Callback):
    def __init__(self,verbose=0):

        super(Callback, self).__init__()
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_loss')
        # if current < 0.014:
        #     self.model.stop_training = True

        predicted = model.predict( matrixTweetsEmb_test )

        test = [ '0' ] * len ( X_test )
        i = 0
        for cl in predicted:
            test[ i ] = str ( np.argmax ( cl ) )
            i += 1

        test_lab = [ '0' ] * len ( X_test )
        i = 0
        for cl in test_classes:
            test_lab[ i ] = str ( np.argmax ( cl ) )
            i += 1

        print ( len ( X_test ) )
        print ( classification_report ( test , test_lab, digits=5 ) )
   


# In[23]:

from keras.layers import *
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

for i in range(0,10):

    model = Sequential ( )
    model.add ( Conv2D ( 200 , (5 , 5) , activation='relu' , input_shape=(100 , 50 , 300) ) )
    model.add ( MaxPool2D ( 2 , 2 ) )
    model.add ( Conv2D ( 100 , (5 , 4) , activation='relu' ) )
    model.add ( MaxPool2D ( 2 , 2 ) )
    model.add ( Conv2D ( 20 , (3 , 3) , activation='relu' ) )
    model.add ( MaxPool2D ( 2 , 2 ) )
    model.add ( Flatten ( ) )
    model.add ( Dense ( 400 , activation="tanh" ) )
    model.add ( Dense ( 200 , activation="tanh" ) )
    model.add ( Dense ( 100 , activation="tanh" ) )
    model.add ( Dense ( 2 , activation="softmax" ) )
    model.summary ( )
    filepath="/Volumes/MacPassport/PycharmProjects/botOrNot/M.MAY.MF.weights.{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}_fasttext_esp_"+str(i)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacks_list = [
        checkpoint,
        MyCallBack(verbose=1)
    ]

    model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )

    history = model.fit(matrixTweetsEmb_train,training_classes,batch_size=32,epochs=10,
                          validation_data= (matrixTweetsEmb_test,test_classes) ,
                          callbacks=callbacks_list,
                          verbose=1)

