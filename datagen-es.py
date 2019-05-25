import configparser
import TwitterAPI
import sys
import pandas as pd
import numpy as np

def get_census_names():
    males = pd.read_csv("male_es.txt",header=None,names=["Name"])
    male_names = np.array(males["Name"])
    male_names = male_names
    females = pd.read_csv("female_es.txt",header=None, names = ["Name"])
    female_names = np.array(females["Name"])
    
    return male_names, female_names


male_names, female_names = get_census_names()

print('Number of Male Names:', len(male_names))
print('Number of Female Names:', len(female_names))
print('male name sample:', list(male_names[:5]))
print('female name sample:', list(female_names[:5]))

def get_twitter(config_file):
    token1, token2, token3, token4 = ''
    config = configparser.ConfigParser()
    config.read(config_file)
    twitter = TwitterAPI.TwitterAPI(token1, token2, token3, token4)
    return twitter

twitter = get_twitter('twitter.cfg')


def get_first_name(tweet):
    if 'user' in tweet and 'name' in tweet['user']:
        parts = tweet['user']['name'].split()
        if len(parts) > 0:
            return parts[0].lower()


def get_first_uid(tweet):
    if 'user' in tweet and 'id_str' in tweet['user']:
        parts = tweet['user']['id_str'].split()
        if len(parts) > 0:
            return parts[0]


def get_first_text(tweet):
    if 'text' in tweet:
        return tweet['text']


def sample_tweets(twitter, limit, male_names, female_names):
    import time
    ids = []
    while True:
        try:
            # Restrict to U.S.
            for response in twitter.request('statuses/filter',{'locations':'-9.868,35.759,2.907,43.018'}):
                time.sleep ( 1 )
                print(response)
                if 'user' in response:
                    name = get_first_name(response)
                    if name in male_names:
                        id = get_first_uid(response)
                        ids.append(id)
                        f = open ( "tweets_es.txt" , "a" )
                        f.write (get_first_uid ( response ) + '\t' + get_first_text ( response ).replace ( '\t' ,' ' ).replace ('\n' , ' ' ).replace ( '\r' , '' ) + '\t' + 'men\n' )
                        f.close ( )

                    if name in female_names:
                        id = get_first_uid ( response )
                        ids.append ( id )
                        f = open ( "tweets_es.txt" , "a" )
                        f.write (get_first_uid ( response ) + '\t' + get_first_text ( response ).replace ( '\t' ,' ' ).replace ('\n' , ' ' ).replace ( '\r' , '' ) + '\t' + 'woman\n' )
                        f.close ( )
        except:
            print("Unexpected error:", sys.exc_info())
    return
        
tweets = sample_tweets(twitter, 100000, male_names, female_names)

print('Number of tweets of users in census list:',len(tweets))

