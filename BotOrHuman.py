import os
import time
import keras
import gensim
import numpy as np
import pandas as pd
import random as rn
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

class BotOrHuman:

    def __init__(self, path, pathToLabels):

        self.path = path
        self.pathToLabels = pathToLabels
        self.botHuman = 3  # Insert 3 to classify bot or human, 4 to classify man or woman

        '''The preprocessor will transform the entities, leaving the smiles,
        removing the stopwords and lemmatizing if necessary'''

        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['email', 'percent', 'money', 'phone',
                       'time', 'url', 'date', 'number'],
            fix_html=True,  # fix HTML tokens
            segmenter="twitter",
            corrector="twitter",
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=True,  # spell correction for elongated words
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )

    def iter_docs(author):

        '''This function extracts the text and the language from the XML'''

        author_attr = author.attrib
        for doc in author.iter('document'):
            doc_dict = author_attr.copy()
            doc_dict.update(doc.attrib)
            doc_dict['data'] = doc.text
            yield doc_dict

    def load_original_dataset(self):

        '''This function load the dataset from a predefined path'''

        start = time.time() # Monitoring time to load the files
        dataEn = pd.DataFrame()

        for root, dirs, files in os.walk(self.path):
            for file in files:
                # Skipping labels
                if file == 'truth.txt':
                    continue
                else:
                    try:
                        pathToFile = root + '/' + file  # Creating path
                        # print(pathToFile) # Just for debugging
                        xml_data = open(pathToFile, "r", encoding="utf8")  # Opening the text file
                        etree = ET.parse(xml_data)  # Create an ElementTree object
                        data = list(BotOrHuman.iter_docs(etree.getroot()))  # Create a list of dictionaries with the data
                        filename = file.split(".")[0]  # Get filename
                        for dictionary in data:  # Loop through the dictionary
                            dictionary['ID'] = filename  # Append filename
                        dataEn = dataEn.append(data)  # Append the list of dictionary to a pandas dataframe

                    # If the file is not valid, skip it
                    except ValueError as e:
                        print(e)
                        continue

        end = time.time()
        print("Total running time to load the dataframe is", end - start)
        return dataEn

    def load_labels(self, data):

        '''This method load the labels of the original dataset'''

        start = time.time()  # Monitoring time to load the files
        target = pd.read_csv(self.pathToLabels, sep=":::", engine='python')
        target.columns = ['ID', 'botOrHuman', 'sex']
        mergedData = pd.merge(data, target, on='ID')
        end = time.time()
        print("Total running time to append the labels to the dataframe is", end - start)
        return mergedData

    def export_dataframe_to_csv(self):

        '''This method exports the dataframe to a csv file to check its consistency
        It is used to check the consistency of the data (debugging)'''

        self.mergedData.to_csv("out.csv", index=False)
        print("Saved the dataframe in a csv file")

    def creating_classes(self, data):

        '''A method to create the list of
        classes, IDs and tweets per each ID'''

        print("Creating classes...")

        listaIds = []
        listaClasses = []
        matrixTweets = []

        try:

            for index, x in data.iterrows():
                id = x['ID']
                if id not in listaIds:
                    newList = list()
                    newList.append(x[1])
                    matrixTweets.append(newList)
                    listaIds.append(id)
                    listaClasses.append(x[self.botHuman])  # Bot or Human
                else:
                    ls = matrixTweets[listaIds.index(id)]
                    ls.append(x[1])
                    matrixTweets[listaIds.index(id)] = ls

            print("Classes created")
            return matrixTweets

        except:
            print('An error occurred.')

    def loading_words_to_vectors(self):

        '''This method loads the Google Word2Vec'''

        print("Loading Google Word2Vec")

        self.google_300 = gensim.models.KeyedVectors.load_word2vec_format(
            "./embeddings/GoogleNews-vectors-negative300.bin", binary=True)

        print("Word2Vec loaded")

    def transforming_sentences_in_vectors(self, matrixTweets):

        '''This method transforms sentences in vectors thanks to the Word2Vec previously loaded'''

        print("Starting the transformation of sentences in vectors..")

        stop_words = set(stopwords.words('english'))
        matrixTweetsEmb = []
        i = 0

        for tweetsUser in matrixTweets:
            embTweetsUser = []
            if (i % 100) == 0:
                print(i)

            for tweet in tweetsUser:
                embTweetUser = np.zeros([50, 300], dtype=np.float16)
                tokList = self.text_processor.pre_process_doc(tweet) # Preprocessing
                tokList = [w for w in tokList if not w in stop_words] # Removing stopwords
                # Find the embeddings
                numTok = 0;
                for token in tokList[0:50]:
                    g_vec = []
                    is_in_model = False
                    if token in self.google_300.vocab.keys():
                        is_in_model = True
                        g_vec = self.google_300.word_vec(token)
                    elif token == "<number>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("number")
                    elif token == "<percent>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("percent")
                    elif token == "<money>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("money")
                    elif token == "<email>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("email")
                    elif token == "<phone>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("phone")
                    elif token == "<time>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("time")
                    elif token == "<date>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("date")
                    elif token == "<url>":
                        is_in_model = True
                        g_vec = self.google_300.word_vec("url")
                    elif not is_in_model:
                        max = len(self.google_300.vocab.keys()) - 1
                        index = rn.randint(0, max)
                        word = self.google_300.index2word[index]
                        g_vec = self.google_300.word_vec(word)

                    embTweetUser[numTok] = np.array(g_vec, dtype=np.float16)
                    numTok += 1
                embTweetsUser.append(np.array(embTweetUser, dtype=np.float16))
            i += 1
            matrixTweetsEmb.append(np.array(embTweetsUser, dtype=np.float16))
            return matrixTweetsEmb

        print("Transformation completed")

    def load_model(self, pathToModel):

        '''This method loads the Keras model
        This will be a two layer classification,
        the first layer will be bot or human,
        the second layer will be man or woman if
        we have not identified a bot'''

        print("Loading model...")
        self.pathToModel = pathToModel

        self.loaded_model_bot_human = keras.models.load_model(self.pathToModel)
        return self.loaded_model_bot_human.summary()

    def make_predictions(self):

        '''This method makes the predictions using the loaded Keras model
        to distinguish between bots and humans'''

        # predictions = self.loaded_model_bot_human.predict(X_test)

        raise NotImplementedError("To be implemented")

    def save_to_xml(self):

        '''This method create an XML with the information of the predictions
        according to the output of the model'''

        # TODO: Replace the defaults with the information predicted from the model
        author = ET.Element("Author")
        ET.SubElement(author, "id").text = "author-id"
        ET.SubElement(author, "lang").text = "en|es"
        ET.SubElement(author, "type").text = "bot|human"
        ET.SubElement(author, "gender").text = "bot|male|female"
        tree = ET.ElementTree(author)
        # TODO: Replace the filename with a loop output - it has to return one file per author of the tweet
        tree.write("xml/filename.xml")
        print("Files saved in the xml directory")

# Instanting the class
start = BotOrHuman(path="./en", pathToLabels="./en/truth.txt") # Replace 'test' with 'en' in production.

# Loading dataset
df = start.load_original_dataset()

# Loading labels to be appended to the dataset
labeled_data = start.load_labels(df)

# Preprocessing data
matrix = start.creating_classes(labeled_data)

# Loading words to vectors binary file
start.loading_words_to_vectors()

# Creating matrices
start.transforming_sentences_in_vectors(matrix)

# Loading models to predict if the user is a bot or a human
start.load_model("./models/02.CNN_100x50x300D_google_0.98.h5")

# Making predictions to understand if the user is a bot or a human
# TODO: complete

# Export the dataframe to csv for debugging purposes
# start.export_dataframe_to_csv()

# Saving predictions to XML
start.save_to_xml()


