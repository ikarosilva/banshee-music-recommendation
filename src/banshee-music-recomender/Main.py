'''
Created on Nov 5, 2015

@author: Ikaro Silva
'''
import sys
from scipy.stats import randint as sp_randint
import argparse
import numpy as np
from numpy import genfromtxt
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from time import time
from operator import itemgetter

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')

def feature_index(feature_names):
        # TODO: Find a way to get this from the db exporter instead
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
        
def train(features,labels): 
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='train recommender', action='store_true')
    args = parser.parse_args()
    
    feature_names=['ArtistID','AlbumID','YEAR','Score','PlayCount','SkipCount']
    db_ind=feature_index(feature_names)
    label_ind=feature_index('Rating')
    print_err("Loading database file (ind=%s)..."%str(db_ind))
    features = genfromtxt('/home/ikaro/music-commands/test', 
                      usecols=db_ind,delimiter=',')
    labels=genfromtxt('/home/ikaro/music-commands/test', 
                      usecols=label_ind,delimiter=',')
    print_err("Finished loading database file. data size=%s"%(features))
    
    classifier=train(features,labels)
    