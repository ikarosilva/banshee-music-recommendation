'''
Created on Nov 5, 2015

@author: Ikaro Silva
'''
import sys
from scipy.stats import randint as sp_randint
import argparse
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from time import time
from operator import itemgetter
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer

HEADER={'PrimarySourceID':"INTEGER NOT NULL",'TrackID':"INTEGER",'ArtistID':"INTEGER",'AlbumID':"INTEGER",'TagSetID':"INTEGER",'ExternalID':"INTEGER",'MusicBrainzID':"TEXT",
                'Uri':"TEXT",'MimeType':"TEXT",'FileSize':"INTEGER",'BitRate':"INTEGER",'SampleRate':"INTEGER",'BitsPerSample':"INTEGER",'Attributes':"INTEGER",
                'LastStreamError':"INTEGER",'Title':"TEXT",'TitleLowered':"TEXT",'TitleSort':"TEXT",'TitleSortKey':"BLOB",'TrackNumber':"INTEGER",'TrackCount':"INTEGER",
                'Disc':"INTEGER",'DiscCount':"INTEGER",'Duration':"INTEGER",'Year':"INTEGER",'Genre':"TEXT",'Composer':"TEXT",'Conductor':"TEXT",'Grouping':"TEXT",
                'Copyright':"TEXT",'LicenseUri':"TEXT",'Comment':"TEXT",'Rating':"INTEGER",'Score':"INTEGER",'PlayCount':"INTEGER",'SkipCount':"INTEGER",
                'LastPlayedStamp':"INTEGER",'LastSkippedStamp':"INTEGER",'DateAddedStamp':"INTEGER",'DateUpdatedStamp':"INTEGER",'MetadataHash':"TEXT",'BPM':"INTEGER",
                'LastSyncedStamp':"INTEGER",'FileModifiedStamp':"INTEGER"
                }

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')
    

def feature_index(feature_names):
        # TODO: Find a way to get this from the db exporter instead
        # To explore database run : cur.execute("select name from sqlite_master where type='table'" )
        train_query="select Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount from CoreTracks where Rating is not null;"
        
            
        return tuple([ i for i, j in enumerate(HEADER) if j in feature_names])

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def get_features(cur,query,header):   
    print_err("Loading database file ...")
    rows=cur.execute(query)
    genre=list()
    composer=list()
    conductor=list()
    genre_vectorizer= CountVectorizer()
    composer_vectorizer= CountVectorizer()
    conductor_vectorizer= CountVectorizer()
    HEADER=header.split(",")
    genre_ind=[ i for i, j in enumerate(HEADER) if j=="Genre"]
    genre_ind=genre_ind[0] if len(genre_ind)>0 else None
    composer_ind=[ i for i, j in enumerate(HEADER) if j=="Composer"]
    composer_ind=composer_ind[0] if len(composer_ind)>0 else None
    conductor_ind=[ i for i, j in enumerate(HEADER) if j=="Conductor"]
    conductor_ind =conductor_ind[0] if len(conductor_ind)>0 else None
    count=0
    columns=len(HEADER)-1
    for row in rows:
        if(genre_ind):
            if(row[genre_ind]):
                genre.append(row[genre_ind])
        if(composer_ind):
            if(row[composer_ind]):
                composer.append(row[composer_ind])
        if(conductor_ind):
            if(row[conductor_ind]):
                conductor.append(row[conductor_ind])
        count+=1
    genre_train = genre_vectorizer.fit_transform(genre)
    composer_train = composer_vectorizer.fit_transform(composer)
    composer_preprocessor=composer_vectorizer.build_preprocessor()
    conductor_train = conductor_vectorizer.fit_transform(conductor)

    #Create feature map
    features=np.zeros((count,columns))
    labels=np.zeros((count,1))
    count=0
    rows=cur.execute(query)
    for row in rows:
        labels=row[0]
        for feature in range(1,columns):
            print("row[%s]=%s"%(feature,row[feature]))
            if(feature == genre_ind and row[feature]):
                features[count,feature]=genre_vectorizer.vocabulary_[row[feature].lower()]
            elif (feature == composer_ind and row[feature]):
                v=composer_preprocessor(row[feature])
                print("v=%s"%(v))
                features[count,feature]=composer_vectorizer.vocabulary_[v]
            elif (feature == conductor_ind and row[feature]):
                features[count,feature]=conductor_vectorizer.vocabulary_[row[feature].lower()]
            else:
                features[count,feature]=row[feature]
        print("features[count,:]=%s"%(features[count,:]))

    print("Generated %s features..."%(str(features.shape)))
    return labels, features, genre_vectorizer, composer_vectorizer,conductor_vectorizer

def train(cur, query,header): 
    labels, features, genre_vectorizer, composer_vectorizer,conductor_vectorizer = get_features(cur,query,header)
    print("features.shape[1,:]=%s"%(features[1,:]))
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": [3, None],
                  "max_features": features.shape[1],
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    
    start = time()
    print("features=%s"%str(np.sum(features)))
    random_search.fit(features,labels)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
        
if __name__ == '__main__':
    
    #r=cur.execute("select sql from sqlite_master where name='CoreTracks'" )
    #r=cur.execute("select TrackID,Rating from CoreTracks where Rating is not null;" )
    #for row in cur:
    #    print row
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='train recommender', action='store_true')
    args = parser.parse_args()
    
    connection = sqlite3.connect('/home/ikaro/.config/banshee-1/banshee.db')
    cur = connection.cursor()
    header="Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount"
    train_query="select " + header +" from CoreTracks where Rating >0;"
    classifier=train(cur,train_query,header)
    