'''
Created on Nov 22, 2015

@author: root
'''
import sys
import re
def genre_map(cur):
    
    stop=['vs', 'ft', 'feat', 'featuring','and','the','faixa','dj','http', ',','/']
    all_genre=[ re.split(r'\W+', i[0].lower().replace("_"," ").replace(","," ").replace("&"," "))
            for i in cur.execute("select Genre from CoreTracks") if i[0] != None and i[0] not in stop]
    all_genre=[re.sub(r'\s+', "", item) for sublist in all_genre for item in sublist]
    genre=list(set(all_genre[:]))
    genre=[ i for i in genre if i not in stop and len(i)>2]
    genre.sort(key=lambda item: (item,-len(item)))
    unique_genre=genre[:]
    remove_ind=list()
    for ind,a in enumerate(unique_genre):
        is_unique = True
        for i in range(ind+1,len(genre)):
            test=genre[i].find(a)
            if(test>=0):
                remove_ind.append(i)
    
    unique_genre=[ i for ind,i in enumerate(unique_genre) if ind not in remove_ind]
    unique_genre=list(set(unique_genre))
    unique_genre.sort(key=lambda item: (item,-len(item)))
    print ("Unique genres= %s"%len(unique_genre))
    genre_map=dict()
    all_genre = [ i[0].lower() for i in cur.execute("select Genre from CoreTracks") if (i[0] != None) ]
    for g_name in all_genre:
        for ind,art in enumerate(unique_genre):
            test=g_name.find(art)
            if(test>=0):
                genre_map[g_name]= ind
                break 
    return genre_map