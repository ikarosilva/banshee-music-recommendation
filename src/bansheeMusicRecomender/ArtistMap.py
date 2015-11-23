'''
Created on Nov 22, 2015

@author: root
'''
import re

def artist_map(cur):
    stop=['vs', 'ft', 'feat', 'featuring','and','the','faixa','dj']
    
    artist=[ " ".join(filter(lambda w: w not in stop, re.split(r'\W+', i[1].lower().replace("_"," ") ))).encode('ascii','ignore') 
            for i in cur.execute("select ArtistId,NameLowered from CoreArtists") if i[1]!= None ]
    
    print (len(artist))
    artist = [ re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s).strip() for s in artist]
    artist = [ " ".join(filter(lambda w: not w.isdigit(), re.split(r'\W+', s))) for s in artist]
    artist=list(set([ i.replace(" ","") for i in artist if len(i)>1]))
    artist.sort(key=lambda item: (item,-len(item)))
    unique_artist=artist[:]
    remove_ind=list()
    for ind,a in enumerate(unique_artist):
        is_unique = True
        for i in range(ind+1,len(artist)):
            test=artist[i].find(a)
            if(test>=0):
                remove_ind.append(ind)
    
    unique_artist=[ i for ind,i in enumerate(unique_artist) if ind not in remove_ind]
    
    all_artists=[i for i in cur.execute("select ArtistId,NameLowered from CoreArtists") if i[1]!= None]
    artist_map=dict()
    for art in all_artists:
        #TODO get index from artist where first string is matched. Key is the artist ID, value is the index of artist.
        pass
    print(len(unique_artist))    
    return unique_artist