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
from numpy import shape

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
    HEADER=header.split(",")
    genre_ind=[ i for i, j in enumerate(HEADER) if j=="Genre"]
    genre_ind=genre_ind[0] if len(genre_ind)>0 else None
    composer_ind=[ i for i, j in enumerate(HEADER) if j=="Composer"]
    composer_ind=composer_ind[0] if len(composer_ind)>0 else None
    conductor_ind=[ i for i, j in enumerate(HEADER) if j=="Conductor"]
    conductor_ind =conductor_ind[0] if len(conductor_ind)>0 else None
    count=0
    columns=len(HEADER)-1
    features=list()
    labels=list()
    empty=0
    for row in rows:
        labels.append(row[0])
        feat_vect=list()
        for feature in range(1,columns):  
            if(row[feature]):
                if(feature == genre_ind):
                    if(row[genre_ind] not in genre):
                        genre.append(row[genre_ind])
                    feat_vect.append(genre.index(row[genre_ind]))
                elif(feature == composer_ind):
                    if(row[composer_ind] not in composer):
                        composer.append(row[composer_ind])
                    feat_vect.append(composer.index(row[composer_ind]))
                elif(feature == conductor_ind):
                    if(row[conductor_ind] not in conductor):
                        conductor.append(row[conductor_ind])
                    feat_vect.append(conductor.index(row[conductor_ind]))
                else:
                    feat_vect.append(row[feature])
            else:
                feat_vect.append(-1)
                empty+=1
        features.append(feat_vect)
        count+=1

    print(" %s rows ( %s empty cells)"%(len(features),empty))
    features=np.array(features)
    labels=np.array(labels)
    print("Generated %s features..."%(str(features.shape)))
    return labels, features, genre,composer,conductor

def train(cur, query,header): 
    labels, features, genre, composer,conductor= get_features(cur,query,header)
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": [3, None],
                  "max_features": [8],
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    
    start = time()
    print("%s features..."%(str(features.shape)))
    print("%s labels..."%((labels)))
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
    