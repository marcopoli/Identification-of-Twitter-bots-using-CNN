import os
import pandas as pd
import numpy as np
#import xml.etree.ElementTree as ET
from lxml import etree as ET
import io
import random as rm

os.listdir()
pathEn = "/Users/kram/Downloads/botOrNot-en_es/en"

def iter_docs(author):
    '''This function extracts the text and the language from the XML'''
    author_attr = author.attrib
    for doc in author.iter('document'):
        doc_dict = author_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict

rm.seed = 75494

#xml_data = open(testfile, "r") # Opening the text file
#etree = ET.parse(xml_data) # Create an ElementTree object
#df = pd.DataFrame(list(iter_docs(etree.getroot()))) #Append the info to a pandas dataframe

tweets = pd.read_csv('/Volumes/MacPassport/PycharmProjects/botOrNot/tweets_es.txt',delimiter="\t",header=None,names= ['id','tweet','gender'])
mens = tweets[tweets.gender == "men"].tweet.tolist()
mensIds = tweets[tweets.gender == "men"].id.tolist()
womans = tweets[tweets.gender == "woman"].tweet.tolist()
womansIds = tweets[tweets.gender == "woman"].id.tolist()

path = "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/truth.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)
humans =  open(path,"w+")

for i in range(0,800):
    author = ET.Element ( "author" )
    author.set ( "lang" , "es" )
    documents = ET.SubElement ( author , "documents" )
    k = 0
    while k < 100:
        index = rm.randint(0,len(mens)-1)
        if len(mens[index]) > 70 and len(mens[index]) < 141:
            doc = ET.SubElement(documents,"document")
            doc.text = ET.CDATA(mens[index])
            mens.pop(index)
            k= k+1
        else:
            mens.pop ( index )

    tree = ET.ElementTree ( author )

    p = "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/" + str(i) + "m.xml"
    os.makedirs ( os.path.dirname ( p ) , exist_ok=True )
    n = open ( "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/" + str(i) + "m.xml",'w+' )
    n.write(ET.tostring(tree,encoding='unicode'))
    n.close();
    humans.write(str(i)+"m:::human:::male\n")

for i in range(0,800):
    author = ET.Element ( "author" )
    author.set ( "lang" , "es" )
    documents = ET.SubElement ( author , "documents" )
    k = 0
    while k < 100:
        index = rm.randint(0,len(womans)-1)
        if len(womans[index]) > 70 and len(womans[index]) < 141:
            doc = ET.SubElement(documents,"document")
            doc.text = ET.CDATA(womans[index])
            #womans.pop(index)
            k = k+1
        else:
            womans.pop(index )

    tree = ET.ElementTree ( author )
    p = "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/" + str(i) + "w.xml"
    os.makedirs ( os.path.dirname ( p ) , exist_ok=True )
    n = open ( "/Volumes/MacPassport/PycharmProjects/botOrNot/genHuman/es/" + str(i) + "w.xml",'w+' )
    n.write(ET.tostring(tree,encoding='unicode'))
    n.close();
    humans.write(str(i)+"w:::human:::female\n")

humans.close()