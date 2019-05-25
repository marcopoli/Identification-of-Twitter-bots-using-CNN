# encoding: utf-8
'''
CS 522 - Advanced Data Mining - Spring 2016 - Final Project
Gender Classification using Twitter Feeds
Step - 1) Collect Surnames from U.S- Census List.
'''

# Import all packages

import requests
import configparser
import TwitterAPI
import sys
from collections import Counter
import pickle

def get_census_names():
    males = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.male.first').text.split('\n')
    females = requests.get('http://www2.census.gov/topics/genealogy/1990surnames/dist.female.first').text.split('\n')

    males_pct = dict([(m.split()[0].lower(), float(m.split()[1])) for m in males if (m != '' and m[0] !='<HTML><HEAD>')])
    females_pct = dict([(f.split()[0].lower(), float(f.split()[1])) for f in females if f])
    male_names = set([m for m in males_pct if m not in females_pct or males_pct[m] > females_pct[m]])
    female_names = set([f for f in females_pct if f not in males_pct or females_pct[f] > males_pct[f]]) 
    return male_names, female_names

male_names, female_names = get_census_names()
print('Number of Male Names:', len(male_names))
print('Number of Female Names:', len(female_names))
print('male name sample:', list(male_names)[:5])
print('female name sample:', list(female_names)[:5])

'''
Conclusion
In 1990 collection of surnames from U.S Census List, there are 1146 Male names and 4014 Female Names.

Step 2 : Sample twitter feeds with names matching census names

Load the Consumer and Access Token Key and Value from twitter.cfg file
'''

def get_twitter(config_file):
    token1, token2, token3, token4 = '' # To be replaced with tokens
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
            for response in twitter.request('statuses/filter',{'locations':'-124.637,24.548,-66.993,48.9974'}):
                time.sleep ( 1 )
                if 'user' in response:
                    name = get_first_name(response)
                    if name in male_names:
                        id = get_first_uid(response)
                        ids.append(id)
                        f = open ( "tweets.txt" , "a" )
                        f.write (get_first_uid ( response ) + '\t' + get_first_text ( response ).replace ( '\t' ,' ' ).replace ('\n' , ' ' ).replace ( '\r' , '' ) + '\t' + 'men\n' )
                        f.close ( )

                    if name in female_names:
                        id = get_first_uid ( response )
                        ids.append ( id )
                        f = open ( "tweets.txt" , "a" )
                        f.write (get_first_uid ( response ) + '\t' + get_first_text ( response ).replace ( '\t' ,' ' ).replace ('\n' , ' ' ).replace ( '\r' , '' ) + '\t' + 'woman\n' )
                        f.close ( )
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return
        
tweets = sample_tweets(twitter, 100000, male_names, female_names)





def junk_tweets(twitter, limit, male_names, female_names):
    junk_tweets = []
    while True:
        try:
            # Restrict to U.S.
            for response in twitter.request('statuses/filter',{'locations':'-124.637,24.548,-66.993,48.9974'}):
                if 'user' in response:
                    name = get_first_name(response)
                    if name not in male_names and name not in female_names:
                        junk_tweets.append(response)
                        if len(junk_tweets) % 100 == 0:
                            print('found %d tweets' % len(junk_tweets))
                        if len(junk_tweets) >= limit:
                            return junk_tweets
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return junk_tweets
        
junk_tweets = junk_tweets(twitter, 200, male_names, female_names)

print('Number of tweets of users in census list:',len(tweets))
print('Number of junk tweets of users not in census list:',len(junk_tweets))
print('Top 10 Most Common Names are: \n', Counter(get_first_name(t) for t in tweets).most_common(10))
print('Top 10 Most Common Junk Names are: \n', Counter(get_first_name(t) for t in junk_tweets).most_common(10))

# Save the normal tweets.
pickle.dump(tweets, open('tweets.pkl', 'wb'))

# Save the junk tweets.
pickle.dump(junk_tweets, open('junk_tweets.pkl', 'wb'))

