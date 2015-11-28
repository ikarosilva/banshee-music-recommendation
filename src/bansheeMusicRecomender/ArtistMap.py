'''
Created on Nov 22, 2015

@author: root
'''
import re

def artist_map(cur):
    stop=['vs', 'ft', 'feat', 'featuring','and','the','faixa','dj', '&']
    seed=['tupac','tiesto','souljaboy']
    artist=[ " ".join(filter(lambda w: w not in stop, re.split(r'\W+', i[1].lower().replace("_"," ").replace(","," ").replace("&"," ") ))).encode('ascii','ignore') 
            for i in cur.execute("select ArtistId,NameLowered from CoreArtists") if i[1]!= None ]
    
    print ("Total artists = %s"%len(artist))
    artist = [ re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s).strip() for s in artist]
    artist = [ re.sub("www.*com", "", s) for s in artist]
    artist = [ re.sub("www.*net", "", s) for s in artist]
    artist = [ " ".join(filter(lambda w: not w.isdigit(), re.split(r'\W+', s))) for s in artist]
    artist=list(set([ i.replace(" ","") for i in artist if len(i)>1]))
    artist.extend(seed)
    artist.sort(key=lambda item: (item,-len(item)))
    unique_artist=artist[:]
    remove_ind=list()
    for ind,a in enumerate(unique_artist):
        is_unique = True
        for i in range(ind+1,len(artist)):
            test=artist[i].find(a)
            if(test>=0):
                remove_ind.append(i)
    
    unique_artist=[ i for ind,i in enumerate(unique_artist) if ind not in remove_ind]
    unique_artist=list(set(unique_artist))
    unique_artist.sort(key=lambda item: (item,-len(item)))
    print("Unique artist=%s"%len(unique_artist))
    all_artists=[i for i in cur.execute("select ArtistId,NameLowered from CoreArtists") if i[1]!= None]
    art_map=dict()
    for query in all_artists:
        for ind,art in enumerate(unique_artist):
            art_name=query[1].lower().strip().replace("_"," ")
            art_name =  re.sub(r'\s+', "", art_name)
            test=art_name.find(art)
            if(test>=0):
                art_map[query[0]]= ind
                break 
    print("Artist map size=%s"%len(art_map.keys()))
    return art_map