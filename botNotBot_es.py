
import os

os.listdir()

pathEn = "/Users/kram/Downloads/botOrNot-en_es/es"

testfile = "/Users/kram/Downloads/botOrNot-en_es/es/1abac92163c7c4a410dad56c8feb7f18.xml"

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

# In[8]:


# Getting ID to insert in the dataframe

filename = testfile.split("/")[-1].split(".")[0]



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


# In[16]:


target = pd.read_csv(pathToLabels, sep=":::")
target.columns=['ID', 'botOrHuman', 'sex']


# We can now proceed with the concatenation of the dataframes for the English language

# In[19]:


mergedEnData = pd.merge(dataEn, target, on='ID')



# # Deep Learning - CNN?

# In[22]:


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

import gc
mergedEnData = None
gc.collect()

'''Trasformo le entità, lascio le faccine, levo le stopword e se serve agli embeddings lemmatizzo'''


# In[23]:


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


# In[25]:


'''Transform sentences to word embeddings'''


# In[26]:


import gensim


# In[27]:


google_300 = gensim.models.KeyedVectors.load_word2vec_format("/Volumes/MacPassport/PycharmProjects/botOrNot/cc.es.300.vec")


# In[33]:


import numpy as np
from nltk.tokenize import TweetTokenizer as TweetTokenizer
from nltk.corpus import stopwords
import random as rn
stop_words = set(stopwords.words('spanish'))
i = 0
matrixTweetsEmb = np.zeros ( (len(matrixTweets) ,100, 50, 300) )
for tweetsUser in matrixTweets:
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
    matrixTweetsEmb[ i ] = np.array ( embTweetsUser )
    i += 1
    embTweetsUser = None
    gc.collect ( )


# In[36]:


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


# In[37]:


#!{sys.executable} -m pip install category_encoders
import category_encoders as ce
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
training_classes = le.fit_transform(listaClasses)
print(le.category_mapping)


# In[ ]:

for i in range(0,10):
    model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )


    from keras.callbacks import ModelCheckpoint
    filepath="/Volumes/MacPassport/PycharmProjects/botOrNot/MAY.weights.{epoch:05d}-{val_loss:.5f}-{val_acc:.5f}_fasttext_esp_"+str(i)+".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


    callbacks_list = [
        checkpoint
    ]

    history = model.fit(matrixTweetsEmb,training_classes,64,7,
                          validation_split= 0.20 ,
                          callbacks=callbacks_list,
                          verbose=1)
