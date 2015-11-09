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

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')

def feature_index(feature_names):
        # TODO: Find a way to get this from the db exporter instead
        # To explore database run : cur.execute("select name from sqlite_master where type='table'" )
        train_query="select Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount from CoreTracks where Rating is not null;"
        HEADER=['PrimarySourceID',#     INTEGER NOT NULL,
                'TrackID',#             INTEGER PRIMARY KEY,
                'ArtistID',#            INTEGER,
                'AlbumID',#             INTEGER,
                'TagSetID',#            INTEGER,
                'ExternalID',#          INTEGER,
                'MusicBrainzID',#       TEXT,
                'Uri',#                 TEXT,
                'MimeType',#            TEXT,
                'FileSize',#            INTEGER,
                'BitRate',#             INTEGER,
                'SampleRate',#          INTEGER,
                'BitsPerSample',#       INTEGER,
                'Attributes',#          INTEGER DEFAULT 5,
                'LastStreamError',#     INTEGER DEFAULT 0,
                'Title',#               TEXT,
                'TitleLowered',#        TEXT,
                'TitleSort',#           TEXT,
                'TitleSortKey',#        BLOB,
                'TrackNumber',#         INTEGER,
                'TrackCount',#          INTEGER,
                'Disc',#                INTEGER,
                'DiscCount',#           INTEGER,
                'Duration',#            INTEGER,
                'Year',#                INTEGER,
                'Genre',#               TEXT,
                'Composer',#            TEXT,
                'Conductor',#           TEXT,
                'Grouping',#            TEXT,
                'Copyright',#           TEXT,
                'LicenseUri',#          TEXT,
                'Comment',#             TEXT,
                'Rating',#              INTEGER,
                'Score',#               INTEGER,
                'PlayCount',#           INTEGER,
                'SkipCount',#           INTEGER,
                'LastPlayedStamp',#     INTEGER,
                'LastSkippedStamp',#    INTEGER,
                'DateAddedStamp',#      INTEGER,
                'DateUpdatedStamp',#    INTEGER,
                'MetadataHash',#        TEXT,
                'BPM',#                 INTEGER,
                'LastSyncedStamp',#     INTEGER,
                'FileModifiedStamp',#   INTEGER
                ]
            
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
        
def train(cur, query): 
    print_err("Loading database file ...")
    rows=cur.execute(query)
    print("Formatting rows into features...")
    genre=list()
    vectorizer= CountVectorizer(min_df=5)
    for row in rows:
        if(row[5]):
            genre.append(row[5])
    x_train = vectorizer.fit_transform(genre)
    N,M=x_train.shape
    print("Processed %s features with %s genres..."%(N,M))
    print(vectorizer.vocabulary_["www"])
    #for gen in vectorizer.get_feature_names():
    #    print(gen)
    SystemExit
    clf = RandomForestClassifier(n_estimators=20)
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    
    start = time()
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
    
    train_query="select Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount from CoreTracks where Rating is not null;"
    classifier=train(cur,train_query)
    