###############################################################
#vers_1 
#videonun metine donusturulmesi, kelimelerin pre-processleri
#kelimelerin sayimi yapilmis. Fakat sonuc elde edilememistir.




#Google API'si 10mb den az dosyalarin sppech-to-text donusumune izin verdiginden
#videonun(.mp4) okunup ses dosyasina donusturulmesi while dongusu icinde 10sn'lik
#parcalar halinde yapilabilir.
import speech_recognition as sr 
import moviepy.editor as mp

clip = mp.VideoFileClip("mL.mp4")
# clip.audio.write_audiofile("converted.wav")


def speech_to_text():
    audio = sr.AudioFile("converted10.wav")
    r = sr.Recognizer()

    with audio as source:
        audio_file = r.record(source)

    result = r.recognize_google(audio_file, language='en-US', show_all = True)
    print(type(result))
    if (type(result) == type([])):
        return ' '
    else:
        return result['alternative'][0]['transcript']
    # with open('recognized.txt',mode ='w') as file: 
    #     file.write("\n") 
    #     if (type(result) == type([])):
    #         file.write('') 
    #     else:
    #         file.write(result['alternative'][0]['transcript']) 
    #     print("ready!")


import math
i=0
res=[]
while i < (math.floor(clip.duration)):
    print(i)
    if (i+10 < clip.duration):
        video = mp.VideoFileClip("mL.mp4").subclip(i,i+10)
        video.audio.write_audiofile("converted10.wav")
    else:
        video = mp.VideoFileClip("mL.mp4").subclip(i,)
        video.audio.write_audiofile("converted10.wav")
    i+=10
    res.append(speech_to_text())
   



###############################################################
##pre-process islemleri
#karakter normalizasyonu
#kisaltilmislarin genisletilmesi
#noktalama isaretlerinin, sayisal ifadelerin kaldirilmasi
#stopwordlerin kaldirilmasi ('like', 'also', 'let', 'lot','hi' sonradan eklenmistir)
#lemmatize islemi
#tek karakterden(harften) olusan kelimelerin kaldirilmasi
#herhangi bir deger icermeyen satirlarin silinmesi

lines = [line.lower() for line in res]

import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"don\'t", "do not", phrase)
    phrase = re.sub(r"doesn\'t", "does not", phrase)
    phrase = re.sub(r"haven\'t", "have not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

noContLines = []
for line in lines:
    noContLines += [decontracted(line)]



noPuncLines = []
for line in noContLines:
    noPuncLines += [re.sub(r'[^\w\s]', ' ', line)]

noNumbLines = []
for line in noPuncLines:
    noNumbLines += [re.sub(r'[0-9]+', ' ', line)]


from gensim.parsing.preprocessing import STOPWORDS
stopwords_gensim = STOPWORDS.union(set(['like', 'also', 'let', 'lot','hi']))
# stopwords_gensim.add("like");
# stopwords_gensim.add("also");
# stopwords_gensim.add("let");
noStopLines= []
for line in noNumbLines:
    noStopLines += [' '.join([word for word in line.split() if not word in stopwords_gensim if len(word)>1])]


from nltk.stem.wordnet import WordNetLemmatizer

noLemmaLines = []
for line in noStopLines:
    noLemmaLines += [' '.join([WordNetLemmatizer().lemmatize(word) for word in line.split() if len(word)>1])]

cleanedLines = list(filter(None, noLemmaLines))

###############################################################


###############################################################
#kelime listesi cikarimi
#satirlarin kelimelerine ayrilmasi

wordList = []
for line in cleanedLines:
    wordList += [word for word in line.split() if len(word)>1]

wordListinLine = []
for line in cleanedLines:
    wordListinLine.append( [word for word in line.split() if len(word)>1])
lengthLine = [len(line) for line in wordListinLine]

###############################################################


###############################################################
#kelimelerin ve 2gramlarin frekanslarinin hesaplanmasi
from nltk.util import ngrams
import collections

ngram_2 = ngrams(wordList, 2)

freqWord = collections.Counter(wordList)
freq2gram = collections.Counter(ngram_2)

###############################################################


###############################################################
#keliemeler ve 2gramlar icin tf-idf hesabi
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_calculator(listLine, ngram):
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(ngram,ngram), use_idf=True, norm='l1')

    tfidf = tfidf_vectorizer.fit_transform(listLine)

    top_sum=tfidf.toarray().sum(axis=0)
    top_sum_tfidf=[top_sum]
    columns_tfidf = tfidf_vectorizer.get_feature_names()


    tf_idfdic = {}
    for i in range(len(top_sum_tfidf[0])):
        tf_idfdic[columns_tfidf[i]]=top_sum_tfidf[0][i]

    return tf_idfdic

tfidf1 = tfidf_calculator(cleanedLines, 1)
tfidf2 = tfidf_calculator(cleanedLines, 2)

###############################################################


###############################################################
#2gramlarin icerdigi kelimeler cinsinden puanlanmasi
#ornek: myScore('machine learning') --> (tf-idf('machine')+tf-idf('learning'))*2gram.Freq('machine learning')
           
import math
scores2gram = dict.fromkeys([' '.join(keys) for keys in freq2gram.keys()],0)
for keys in freq2gram.keys():
    scores2gram[' '.join(keys)] = math.log10(tfidf1[keys[0]]+tfidf1[keys[1]])*freq2gram[keys]


import operator

def sort_dict2List(dictScores):
    sortedDict=sorted(dictScores.items(),reverse=True,key=operator.itemgetter(1))
    return sortedDict[0:10]


sortedTFIDF1Scores = sort_dict2List(tfidf1)
sortedTFIDF2Scores = sort_dict2List(tfidf2)
sortedScores2gramScores = sort_dict2List(scores2gram)

###############################################################


###############################################################
#satirlar cumle yapisina bakilmadan olusturuldugundan her satirin sonuna
#bir sonraki satirin ilk kelimesi eklenir
mergeLines = []
for i in range(len(cleanedLines)-1):
    mergeLines += [cleanedLines[i] + " " +cleanedLines[i+1].split()[0]]
mergeLines += [cleanedLines[i+1]]

###############################################################


###############################################################
#hangi satirlarin hangi top20 kelime/kelimedizisini icerdigini
#gostermek icin dataframe olusturdum, fakat ilerleyen asamalarda bunu 
#kullanmiyorum. Burada amacim yer tespitine gore baslik secimi
#yapabilecegimi dusunmemdi.
from pandas import DataFrame
df = DataFrame(mergeLines, columns=['cleaned lines'])


def create_vector(dFrame, sortedScores):
    zeros = [0]*len(dFrame)
    top10 = []
    for i in range (len(sortedScores)):
        top10.append(''.join(sortedScores[i][0]))
        dFrame[top10[i]] = zeros 
        
  

create_vector(df, sortedTFIDF1Scores)
create_vector(df, sortedTFIDF2Scores)
create_vector(df, sortedScores2gramScores)


top10 = []
for i in range (len(sortedTFIDF1Scores)):
    top10.append(sortedTFIDF1Scores[i][0])

for i in range (len(sortedTFIDF2Scores)):
    top10.append(sortedTFIDF2Scores[i][0])

for i in range (len(sortedScores2gramScores)):
    top10.append(sortedScores2gramScores[i][0])


for i in range (len(df)):
    for k in range(len(top10)):
        if (' '+top10[k]+' ') in (''+df['cleaned lines'].loc[i]):
            df[top10[k]].loc[i] = k+1
        else:
            df[top10[k]].loc[i] = 0

###############################################################


###############################################################
#herbir cumlenin icerdigi en yuksek skorlu kelime/kelimedizilerine
#gore puanlandirilmasi saglanir. siniflandirma islemi buradan alinacak
#degerlere gore yapilmasi gerekirdi.
def sentence_score(mergeSentences,score1,score2,score3):
    sentenceScore = []
    counter =0
    for sent in mergeSentences:
        sentenceScore.append(0)
        i=0
        for i in range (len(score1)):
            if(sent.find(score1[i][0])>=0):
                sentenceScore[counter]+= score1[i][1]
        i=0
        for i in range (len(score2)):
            if(sent.find(score2[i][0])>=0):
                sentenceScore[counter]+= score2[i][1]
        i=0
        for i in range (len(score3)):
            if(sent.find(score3[i][0])>=0):
                sentenceScore[counter]+= score3[i][1]
        counter+=1
    return sentenceScore

sentScore = sentence_score(mergeLines,sortedTFIDF1Scores,
                           sortedTFIDF2Scores,sortedScores2gramScores)

###############################################################


from sklearn.cluster import KMeans
# features = df.drop(['cleaned lines'], axis=1)
df2 = DataFrame(sentScore, columns=['score'])
df2['zeros'] = [0]*len(sentScore)
kmeans = KMeans(n_clusters=5, random_state=0).fit(df2)
print(kmeans.labels_)

