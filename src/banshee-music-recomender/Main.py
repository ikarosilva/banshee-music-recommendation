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

CLASS_CUTOFF=4
LIMIT=8000

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')
    

def feature_index(feature_names):
        # TODO: Find a way to get this from the db exporter instead
        # To explore database run : cur.execute("select name from sqlite_master where type='table'" )
        train_query="select Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount from CoreTracks where Rating is not null;"
        
            
        return tuple([ i for i, j in enumerate(HEADER) if j in feature_names])

def remove_repeats(features,labels,uri):
        print("Searching for repeated features")
        repeats=set()
        # ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount,Uri"
        col_ind=range(features.shape[1])
        M = len(col_ind)
        remove_uri=list()
        for i in range(np.shape(features)[0]-1):
            for j in range(i+1,np.shape(features)[0]):
                not_repeat=0
                for col in col_ind:
                    if(features[i,col]==features[j,col]):
                        not_repeat+=1
                if(not_repeat == M ):#and (features[i,0] != 10351)):
                    title1=uri[i].split('/')
                    title2=uri[j].split('/')
                    if(title1[-1]==title2[-1]):
                        #print("features[%s,col]=%s"%(i,features[i,:]))
                        #print("features[%s,col]=%s"%(j,features[j,:]))
                        #print("%s"%(str(uri[i])))
                        #print("%s"%(str(uri[j])))
                        repeats.add(j)
                        remove_uri.append(uri[j])
            
        print("Removing %s repeated rows"%(len(repeats)))
        features=np.delete(features, list(repeats), axis=0)
        if (labels != None):
            labels=np.delete(labels,list(repeats),axis=0)
        uri=[x for x in uri if x not in remove_uri]
        return features,labels,uri

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
    uri=list()
    HEADER=header.split(",")
    genre_ind=[ i for i, j in enumerate(HEADER) if j=="Genre"]
    genre_ind=genre_ind[0] if len(genre_ind)>0 else None
    composer_ind=[ i for i, j in enumerate(HEADER) if j=="Composer"]
    composer_ind=composer_ind[0] if len(composer_ind)>0 else None
    conductor_ind=[ i for i, j in enumerate(HEADER) if j=="Conductor"]
    conductor_ind =conductor_ind[0] if len(conductor_ind)>0 else None
    uri_ind=[ i for i, j in enumerate(HEADER) if j=="Uri"]
    uri_ind =uri_ind[0] if len(uri_ind)>0 else None
    count=0
    columns=len(HEADER)-1
    features=list()
    labels=list()
    test_features=list()
    test_uri=list()
    empty=0
    for row in rows:
        if(row[0]==0):
            in_test=True
        else:
            in_test=False
           
        feat_vect=list()
        for feature in range(1,columns+1):  
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
                elif(feature == uri_ind):
                    if(in_test):
                        test_uri.append(row[uri_ind])
                    else:
                        uri.append(row[uri_ind])
                else:
                    feat_vect.append(row[feature])
            else:
                feat_vect.append(-1)
                empty+=1
                #print("feat=%s"%(feat_vect))
                #print("row=%s"%(str(row)))
        if(in_test):
            test_features.append(feat_vect)
        else:
            labels.append(row[0])
            features.append(feat_vect)
            count+=1
            if(count>LIMIT):
                break

    print(" %s rows ( %s empty cells) (uri=%s)"%(len(features),empty,len(uri)))
    features=np.array(features)
    test_features=np.array(test_features)
    
    labels=[1 if x>CLASS_CUTOFF else 0 for x in labels]
    labels=np.array(labels)
    features,labels,uri=remove_repeats(features,labels,uri)
    test_features,_,test_uri=remove_repeats(test_features,None,test_uri)
    print("Generated %s features..."%(str(features.shape)))
    return labels, features, genre,composer,conductor, uri, test_features, test_uri

def train(cur, query,header): 
    labels, features, genre, composer,conductor, uri, test_features, test_uri= get_features(cur,query,header)
    
    
    clf = RandomForestClassifier()
    param_dist = {"max_depth": [8, 16],
                  "max_features": [2, 4 , 8],
                  "min_samples_split": [1, 3, 5, 10],
                  "min_samples_leaf": [3 , 11, 16],
                  "bootstrap": [False],
                  "n_estimators":[2, 5, 20, 40, 80],
                  "criterion": ["gini"]}
    
    # run randomized search
    n_iter_search = 100
    random_search = GridSearchCV(clf,param_grid=param_dist,n_jobs=3, refit=True)
    start = time()
    print("Training classifier...")
    print("%s features..."%(str(features.shape)))
    #print("%s labels..."%((labels)))
    random_search.fit(features,labels)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
    weight=forest.feature_importances_[:].sort()
    feat_names=header[1:-1] #ignore first (label) and last (uri)
    best_features=[feat_names[forest.feature_importances_[i]._index] for i in weight ]
    print("Best features=%s"%str(best_features))
    print("Generating suggestion list from %s songs"%(len(test_uri)))
    predictions= random_search.predict(features)
    suggestions=[ test_uri[i] for i,x in enumerate(predictions) if x==1]
    for song in suggestions:
        print "Suggestion: "+ str(song)
    
if __name__ == '__main__':
    
    #r=cur.execute("select sql from sqlite_master where name='CoreTracks'" )
    #r=cur.execute("select TrackID,Rating from CoreTracks where Rating is not null;" )
    #for row in cur:
    #    print row
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='train recommender', action='store_true')
    args = parser.parse_args()
    start = time()
    connection = sqlite3.connect('/home/ikaro/.config/banshee-1/banshee.db')
    cur = connection.cursor()
    header="Rating, ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount,Uri"
    train_query="select " + header +" from CoreTracks;"
    classifier=train(cur,train_query,header)
    print("Classification complete, total time= %s minutes" % (str((time() - start)/60.0)))