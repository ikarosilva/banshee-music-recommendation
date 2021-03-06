#/usr/bin/env python
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
from gensim import corpora, models
from ArtistMap import artist_map
from GenreMap import genre_map
import re
from functools import wraps

HEADER={'PrimarySourceID':"INTEGER NOT NULL",'TrackID':"INTEGER",'ArtistID':"INTEGER",'AlbumID':"INTEGER",'TagSetID':"INTEGER",'ExternalID':"INTEGER",'MusicBrainzID':"TEXT",
                'Uri':"TEXT",'MimeType':"TEXT",'FileSize':"INTEGER",'BitRate':"INTEGER",'SampleRate':"INTEGER",'BitsPerSample':"INTEGER",'Attributes':"INTEGER",
                'LastStreamError':"INTEGER",'Title':"TEXT",'TitleLowered':"TEXT",'TitleSort':"TEXT",'TitleSortKey':"BLOB",'TrackNumber':"INTEGER",'TrackCount':"INTEGER",
                'Disc':"INTEGER",'DiscCount':"INTEGER",'Duration':"INTEGER",'Year':"INTEGER",'Genre':"TEXT",'Composer':"TEXT",'Conductor':"TEXT",'Grouping':"TEXT",
                'Copyright':"TEXT",'LicenseUri':"TEXT",'Comment':"TEXT",'Rating':"INTEGER",'Score':"INTEGER",'PlayCount':"INTEGER",'SkipCount':"INTEGER",
                'LastPlayedStamp':"INTEGER",'LastSkippedStamp':"INTEGER",'DateAddedStamp':"INTEGER",'DateUpdatedStamp':"INTEGER",'MetadataHash':"TEXT",'BPM':"INTEGER",
                'LastSyncedStamp':"INTEGER",'FileModifiedStamp':"INTEGER"
                }

CLASS_CUTOFF=3
LIMIT=8000

def timefn(fn):
    @wraps(fn)
    def measure_time(*args,**kwargs):
        t1 = time.time()
        results = fn(*args,**kwargs)
        t2 = time.time()
        pritn("@timefn:"+ fn.func_name + " processing time: " + str(t2-t1) +  " seconds")
        return results
    return measure_time

def print_err(*args):
    sys.stderr.write(' '.join(map(str,args)) + '\n')
   
def add_suggestion(cur,connection,track_id):
    rows=cur.execute("select max(EntryId) from CorePlaylistEntries")
    #print("\n".join([str(i) for i in rows]))
    index=rows.fetchall()[0][0]+1
    order=cur.execute("select max(ViewOrder) from CorePlaylistEntries where PlaylistID=13")
    view_order=order.fetchall()[0][0]+1
    query="insert into CorePlaylistEntries (EntryID,PlaylistID,TrackID,ViewOrder,Generated) values (%s,13,%s,%s,0)"%(index,track_id,view_order)
    print query
    cur.execute(query)
    connection.commit()
    print("Inserted suggestion into db.Out:")

def feature_index(feature_names):
        # TODO: Find a way to get this from the db exporter instead
        # To explore database run : cur.execute("select name from sqlite_master where type='table'" )
        train_query="select Rating,ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,PlayCount,SkipCount from CoreTracks where Rating is not null;"
        
            
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
    
    print ("Getting genre info...")
    unique_genre_map=genre_map(cur)
    print("Genres=%s"%str(unique_genre_map.keys()))
    #Get artist map, to clean up redundancies
    print("Getting artist info...")
    unique_artists_map=artist_map(cur)
    print("Loading database file ...")
    rows=cur.execute(query)
    genre=list()
    composer=list()
    conductor=list()
    uri=list()
    title=list()
    test_title=list()
    HEADER=header.split(",")
    genre_ind=[ i for i, j in enumerate(HEADER) if j=="Genre"]
    genre_ind=genre_ind[0] if len(genre_ind)>0 else None
    artist_ind=[ i for i, j in enumerate(HEADER) if j=="ArtistID"]
    artist_ind=artist_ind[0] if len(artist_ind)>0 else None
    composer_ind=[ i for i, j in enumerate(HEADER) if j=="Composer"]
    composer_ind=composer_ind[0] if len(composer_ind)>0 else None
    conductor_ind=[ i for i, j in enumerate(HEADER) if j=="Conductor"]
    conductor_ind =conductor_ind[0] if len(conductor_ind)>0 else None
    uri_ind=[ i for i, j in enumerate(HEADER) if j=="Uri"]
    uri_ind =uri_ind[0] if len(uri_ind)>0 else None
    title_ind=[ i for i, j in enumerate(HEADER) if j=="Title"]
    title_ind =title_ind[0] if len(title_ind)>0 else None
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
                    #Clean up ID, mapp do less redundant set
                    tmp_genre=row[feature].lower().encode('ascii','ignore') #.replace("_"," ").replace(" ","").strip()
                    #tmp_genre = re.sub(r'\s+', "", tmp_genre) 
                    genre_id=unique_genre_map.get(tmp_genre,-1)
                    if(genre_id==-1):
                        #print sorted(unique_genre_map.keys())
                        unique_genre_map[tmp_genre]=-2
                        print("Could not find key for genre:%s**"%(tmp_genre))
                        #print unique_genre_map.keys()
                        #sys.exit(0)
                    feat_vect.append(genre_id)
                elif(feature == composer_ind):
                    if(row[composer_ind] not in composer):
                        composer.append(row[composer_ind].lower())
                    feat_vect.append(composer.index(row[composer_ind].lower()))
                elif(feature == conductor_ind):
                    if(row[conductor_ind] not in conductor):
                        conductor.append(row[conductor_ind].lower())
                    feat_vect.append(conductor.index(row[conductor_ind].lower()))
                elif(feature == title_ind):
                    if(in_test):
                        test_title.append(row[title_ind].lower().split())
                    else:
                        title.append(row[title_ind].lower().split())
                elif(feature == uri_ind):
                    if(in_test):
                        test_uri.append(row[uri_ind])
                    else:
                        uri.append(row[uri_ind])
                elif(feature == artist_ind):
                    #Clean up ID, mapp do less redundant set
                    artist_id=unique_artists_map.get(row[feature],-1)
                    if(artist_id==-1):
                        unique_artists_map[row[feature]]=-2
                        print("Could not find key for artist:%s"%(row[feature]))
                        #print unique_artists_map.keys()
                        #sys.exit(0)
                    feat_vect.append(artist_id)
                else:
                    feat_vect.append(row[feature])
            else:
                feat_vect.append(-1)
                empty+=1
                #print("feat=%s"%(feat_vect))
        if(in_test):
            test_features.append(feat_vect)
        else:
            labels.append(row[0])
            features.append(feat_vect)
            count+=1
            if(count>LIMIT):
                break

    #Get title topics
    #print(title)
    #model = models.ldamodel.LdaModel(title,num_topics=5)
    #print(model[title]) 
    print(" %s rows ( %s empty cells) (uri=%s)"%(len(features),empty,len(uri)))
    features=np.array(features)
    test_features=np.array(test_features)
    
    labels=[1 if x>CLASS_CUTOFF else 0 for x in labels]
    labels=np.array(labels)
    features,labels,uri=remove_repeats(features,labels,uri)
    test_features,_,test_uri=remove_repeats(test_features,None,test_uri)
    print("Generated %s features..."%(str(features.shape)))
    return labels, features, genre,composer,conductor, uri, test_features, test_uri

def train_svm(cur, query,header): 
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    labels, features, genre, composer,conductor, uri, test_features, test_uri= get_features(cur,query,header)
    

    # Split the dataset in two equal parts
    sc = StandardScaler().fit(features,labels)
    #features = sc.transform(features,labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                 'C': [1, 10]},
    #                {'kernel': ['linear'], 'C': [1, 10]}]

    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10]}]

    scores = ['precision', 'recall']

    start = time()
    clf=None
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,n_jobs=3, 
                           scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    
    
    print("Generating suggestion list from %s songs"%(len(test_uri)))
    #test_features = sc.transform(test_features)
    predictions= clf.predict(test_features)
    suggestions=[ test_uri[i] for i,x in enumerate(predictions) if x==1]
    #suggestions=[ uri[i] for i,x in enumerate(predictions) if x==1]
    f = open('/tmp/recommender.m3u','w')
    print("%s suggestions available."%(len(suggestions)))
    for song in suggestions:
        #print "Suggestion: "+ str(song)
        f.write(str(song)+'\n')
    f.close()
    
def train_forest(cur, query,header): 
    from sklearn.preprocessing import StandardScaler
    labels, features, genre, composer,conductor, uri, test_features, test_uri= get_features(cur,query,header)
    
    sc = StandardScaler().fit(features,labels)
    #features = sc.transform(features,labels)
    clf = RandomForestClassifier()
    param_dist = {"max_depth": [4, 8, 16],
                  "max_features": [2, 4 , np.shape(features)[1]],
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
    weight=random_search.best_estimator_.feature_importances_[:].tolist()
    weight.sort(reverse=True)
    original=random_search.best_estimator_.feature_importances_[:].tolist()
    weight.sort()
    feat_names=header.split(',')
    feat_names=feat_names[1:-1] #ignore first (label) and last (uri)
    best_features=[feat_names[original.index(i)] for i in weight ]
    print("Best features=%s"%str(best_features))
    print("Generating suggestion list from %s songs"%(len(test_uri)))
    #test_features = sc.transform(test_features)
    predictions= random_search.predict(test_features)
    suggestions=[ test_uri[i] for i,x in enumerate(predictions) if x==1]
    f = open('/tmp/recommender.m3u','w')
    print("%s suggestions available."%(len(suggestions)))
    for song in suggestions:
        #print "Suggestion: "+ str(song)
        f.write(str(song)+'\n')
    f.close()
    
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
    #header="Rating,ArtistID,AlbumID,Disc,Year,Genre,Composer,Conductor,Duration,Uri"
    header="Rating,ArtistID,Genre,Uri"
    train_query="select " + header +" from CoreTracks;"
    classifier=train_svm(cur,train_query,header)
    print("Classification complete, total time= %s minutes" % (str((time() - start)/60.0)))
    connection.close()