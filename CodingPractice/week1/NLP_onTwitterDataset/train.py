import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#----------READING THE CSV FILE THROUGH PANDAS
train = pd.read_csv('train_E6oV3lV.csv')

#print(train)

#----------FETCHING OUT THE WORD COUNT
train['word_count']= train['tweet'].apply(lambda x: len(str(x).split(" ")))
#print(train[['tweet','word_count']].head())

#----------FETCHING OUT THE CHARACTER COUNT
train['char_count'] = train['tweet'].str.len()
#print(train[['tweet','char_count']].head())

#----------FETCHING OUT THE AVERAGE WORD COUNT
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

train['avg_word']=train['tweet'].apply(lambda x: avg_word(x))
print(train[['tweet', 'avg_word']].head())


#---------FETCHING OUT NO. OF STOP WORDS
stop=stopwords.words('english')

train['stopwords']=train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
#print(train[['tweet','stopwords']].head())


#---------FETCHING OUT THE HASHTAGS AND MENTIONS
train['hashtags']=train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
#print(train[['tweet','hashtags']].head())


#---------FETCHING OUT THE NUMBER OF NUMERICS
train['numerics']= train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
#print(train[['tweet','numerics']].head())


#---------FETCHING OUT THE NUMBER OF UPPERCASE WORDS
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
#print(train[['tweet','upper']].head())


#---------LOWER CASING
train['tweet']=train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#print(train['tweet'].head())


#---------REMOVING PUNCTUATION
train['tweet']=train['tweet'].str.replace('[^\w\s]','')
#print(train['tweet'].head())


#---------REMOVAL OF STOP WORDS
#train['tweet']=train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#print(train['tweet'].head())


#---------COMMON WORD REMOVAL
freq=pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
#print(freq)
freq=list(freq.index)
#train['tweet']=train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#print(train['tweet'].head())


#---------RARE WORDS REMOVAL
freq=pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
#print(freq)
freq=list(freq.index)
#train['tweet']=train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#print(train['tweet'].head())


#---------SPELLING CORRECTION
#print(train['tweet'][:10].apply(lambda x: str(TextBlob(x).correct())))


#----------TOKENIZATION
#print(TextBlob(train['tweet'][1]).words)


#----------STEMMING
st=PorterStemmer()
#print(train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()])))


#----------LEMMATIZATION
#train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#print(train['tweet'].head())



#----------N-GRAMS
#print(TextBlob(train['tweet'][0]).ngrams(2))


#----------TERM FREQUENCY
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf1.columns = ['words', 'tf']
#print(tf1)


#----------INVERSE DOCUMENT FREQUENCY
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))
#print(tf1)



#----------TERM FREQUENCY-INVERSE DOCUMENT FREQUENCY(TF-IDF)
tf1['tfidf']=tf1['tf']*tf1['idf']
#print(tf1)

#----------SKLEARN TF_IDF
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])

#print(train_vect)


#----------BAG OF WORDS
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
#print(train_bow)


#---------SENTIMENT ANALYSIS
#print(train['tweet'][:5].apply(lambda x: TextBlob(x).sentiment))

train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
#print(train[['tweet','sentiment']].head())




