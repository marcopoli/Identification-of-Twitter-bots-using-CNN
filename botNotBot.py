import os
import numpy as np
from sklearn.metrics import accuracy_score
os.listdir()
pathEn = "en"
import joblib
matrixTweetsEmb = joblib.load('matrixTweetsEmb_ALT.dump')
listaClasses = joblib.load('listaClasses.dump')
#
# import resource
#
# def set_memory_limit(memory_kilobytes):
#     # ru_maxrss: peak memory usage (bytes on OS X, kilobytes on Linux)
#     usage_kilobytes = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#     rlimit_increment = 1024 * 1024
#     resource.setrlimit(resource.RLIMIT_DATA, (rlimit_increment, resource.RLIM_INFINITY))
#
#     memory_hog = []
#
#     while usage_kilobytes() < memory_kilobytes:
#         try:
#             for x in range(100):
#                 memory_hog.append('x' * 400)
#         except MemoryError as err:
#             rlimit = resource.getrlimit(resource.RLIMIT_DATA)[0] + rlimit_increment
#             resource.setrlimit(resource.RLIMIT_DATA, (rlimit, resource.RLIM_INFINITY))
#
# set_memory_limit(15000 * 1024)  # 50 mb
#
# # First of all, we will create a procedure to be tested on a single file.
# #
# # After this first step will be completed, we will extend this procedure to create a complete dataframe
#
# testfile = "en/1a5b808546838869bc39cebdbad951e3.xml"
#
# import pandas as pd
# import xml.etree.ElementTree as ET
# import io
#
# def iter_docs(author):
#     '''This function extracts the text and the language from the XML'''
#     author_attr = author.attrib
#     for doc in author.iter('document'):
#         doc_dict = author_attr.copy()
#         doc_dict.update(doc.attrib)
#         doc_dict['data'] = doc.text
#         yield doc_dict
#
# xml_data = open(testfile, "r") # Opening the text file
# etree = ET.parse(xml_data) # Create an ElementTree object
# df = pd.DataFrame(list(iter_docs(etree.getroot()))) #Append the info to a pandas dataframe
# df.head(10)
# df['data'][0]
# filename = testfile.split("/")[-1].split(".")[0]
#
# import time
#
# # Creating empty dataframe
# dataEn = pd.DataFrame()
#
# # Monitoring time to load the files
# start = time.time()
#
# for root, dirs, files in os.walk(pathEn):
#     for file in files:
#         if file == 'truth.txt':
#             continue
#         else:
#             try:
#                 pathToFile = root + '/' + file # Creating path
#                 # print(pathToFile) # Just for debugging
#                 xml_data = open(pathToFile, "r", encoding="utf8") # Opening the text file
#                 etree = ET.parse(xml_data) # Create an ElementTree object
#                 data = list(iter_docs(etree.getroot())) # Create a list of dictionaries with the data
#                 filename = file.split(".")[0] # Get filename
#                 for dictionary in data: # Loop through the dictionary
#                     dictionary['ID'] = filename # Append filename
#                 dataEn = dataEn.append(data)  # Append the list of dictionary to a pandas dataframe
#
#             # If the file is not valid, skip it
#             except ValueError as e:
#                 print(e)
#                 continue
#
# end = time.time()
# print("Total running time is", end - start)
#
# dataEn.head(10)
#
# dataEn['data'][0]
# dataEn.describe()
#
# pathToLabels = "en/truth.txt"
#
# target = pd.read_csv(pathToLabels, sep=":::")
# target.columns=['ID', 'botOrHuman', 'sex']
# target.head(10)
# target.describe()
# mergedEnData = pd.merge(dataEn, target, on='ID')
# mergedEnData.head(10)
# mergedEnData.describe()
#
#
# # # Deep Learning - CNN?
# '''Creo la litsa degli ID, delle classi e dei tweets pr ogni ID'''
# listaIds = [ ]
#
# matrixTweets = [ ]
#
# for index , x in mergedEnData.iterrows ( ):
#     id = x[ 'ID' ]
#     if id not in listaIds:
#         newList = list ( )
#         newList.append ( x[ 1 ] )
#         matrixTweets.append ( newList )
#         listaIds.append ( id )
#         listaClasses.append ( x[ 3 ] )
#     else:
#         ls = matrixTweets[ listaIds.index ( id ) ]
#         ls.append ( x[ 1 ] )
#         matrixTweets[ listaIds.index ( id ) ] = ls
#
# print ( len ( listaIds ) )

# '''Trasformo le entità, lascio le faccine, levo le stopword e se serve agli embeddings lemmatizzo'''
#
# from ekphrasis.dicts.emoticons import emoticons
# from ekphrasis.classes.preprocessor import TextPreProcessor
# from ekphrasis.classes.tokenizer import SocialTokenizer
#
# text_processor = TextPreProcessor (
#     # terms that will be normalized
#     normalize=[ 'email' , 'percent' , 'money' , 'phone' ,
#                 'time' , 'url' , 'date' , 'number' ] ,
#     fix_html=True ,  # fix HTML tokens
#     segmenter="twitter" ,
#     corrector="twitter" ,
#     unpack_hashtags=True ,  # perform word segmentation on hashtags
#     unpack_contractions=True ,  # Unpack contractions (can't -> can not)
#     spell_correct_elong=True ,  # spell correction for elongated words
#     tokenizer=SocialTokenizer(lowercase=True).tokenize,
#     dicts=[ emoticons ]
# )

# import gensim
# google_300 = gensim.models.KeyedVectors.load_word2vec_format( "google_w2v_300.bin" , binary=True )
#
# from nltk.tokenize import TweetTokenizer as TweetTokenizer
# from nltk.corpus import stopwords
# import random as rn
# import nltk
# nltk.download('stopwords')
# stop_words = set ( stopwords.words ( 'english' ) )
# numUs = len(listaIds)
# i = 0
# matrixTweetsEmb = [ ]
# for tweetsUser in matrixTweets:
#     if(i % 100) == 0:
#         print(i)
#     embTweetsUser = [ ]
#
#     for tweet in tweetsUser:
#         embTweetUser = np.zeros ( [ 50 , 300 ] )
#         # Preprocesso
#         tokList = text_processor.pre_process_doc ( tweet )
#         # Rimuovo le stopwords
#         tokList = [ w for w in tokList if not w in stop_words ]
#         # trovo l'embedding
#         numTok = 0;
#         for token in tokList[ 0:50 ]:
#             g_vec = [ ]
#             is_in_model = False
#             if token in google_300.vocab.keys ( ):
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( token )
#             elif token == "<number>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "number" )
#             elif token == "<percent>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "percent" )
#             elif token == "<money>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "money" )
#             elif token == "<email>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "email" )
#             elif token == "<phone>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "phone" )
#             elif token == "<time>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "time" )
#             elif token == "<date>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "date" )
#             elif token == "<url>":
#                 is_in_model = True
#                 g_vec = google_300.word_vec ( "url" )
#             elif not is_in_model:
#                 max = len ( google_300.vocab.keys ( ) ) - 1
#                 index = rn.randint ( 0 , max )
#                 word = google_300.index2word[ index ]
#                 g_vec = google_300.word_vec ( word )
#
#             embTweetUser[ numTok ] = np.array ( g_vec )
#             numTok += 1
#         embTweetsUser.append ( np.array ( embTweetUser ) )
#
#     matrixTweetsEmb.append ( np.array ( embTweetsUser ) )
#     i += 1
#
# import numpy as np
# matrixTweetsEmb = np.array(matrixTweetsEmb)
# print(matrixTweetsEmb.shape)

#get_ipython().system('{sys.executable} -m pip install joblib')




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

#get_ipython().system('{sys.executable} -m pip install category_encoders')
import category_encoders as ce
le =  ce.OneHotEncoder(return_df=False, impute_missing=False, handle_unknown="ignore")
training_classes = le.fit_transform(listaClasses)
print(le.category_mapping)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(matrixTweetsEmb,listaClasses, test_size=0.15, random_state=891)

y_test = le.transform(y_test)

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

        predicted = model.predict ( X_test )

        test = [ '0' ] * len ( X_test )
        i = 0
        for cl in predicted:
            test[ i ] = str ( np.argmax ( cl ) )
            i += 1

        test_lab = [ '0' ] * len ( X_test )
        i = 0
        for cl in y_test:
            test_lab[ i ] = str ( np.argmax ( cl ) )
            i += 1

        print ( len ( X_test ) )
        acc = accuracy_score ( test , test_lab )
        print ( "Accuracy:" , acc )
        print ( classification_report ( test , test_lab ) )
        
callbacks_list = [
    MyCallBack(verbose=1)
]

model.compile ( loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'] )

from sklearn.model_selection import train_test_split, StratifiedKFold



folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=7654).split(X_train,y_train))
X_tr = []
y_tr = []
X_te = []
y_te = []
y_train = le.transform(y_train)
for j , (train_idx , val_idx) in enumerate ( folds ):
    print ( '\nFold ' , j )
    X_tr = X_train[ train_idx ]
    y_tr = y_train[ train_idx ]
    X_te = X_train[ val_idx ]
    y_te = y_train[ val_idx ]

    history = model.fit(X_tr,y_tr,64,5,
                          validation_data= (X_te,y_te) ,
                          callbacks=callbacks_list,
                          verbose=1)


predicted = model.predict ( X_test )

test = [ '0' ] * len ( X_test )
i = 0
for cl in predicted:
    test[ i ] = str ( np.argmax ( cl ) )
    i += 1

test_lab = [ '0' ] * len ( X_test )
i = 0
for cl in y_test:
    test_lab[ i ] = str ( np.argmax ( cl ) )
    i += 1

print ( len ( X_test ) )
acc = accuracy_score(test, test_lab)
print("Accuracy:", acc)
print ( classification_report ( test , test_lab ) )

model.save('02.CNN_100x50x300D_google.h5')