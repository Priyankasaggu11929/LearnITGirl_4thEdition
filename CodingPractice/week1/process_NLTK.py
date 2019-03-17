from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


#----load data----
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

#----split into words----
tokens = word_tokenize(text)
#print(tokens[:100])
#print(len(tokens))

#----conver into lower case----
#tokens=[w.lower() for w in tokens]

#----remove punctuation from each word----
#table=str.maketrans('','', string.punctuation)
#stripped = [w.translate(table) for w in tokens]

#----split into sentences----
#sen=sent_tokenize(text)
#print(sen[:100])

#----remove all tokens that are not alphabetic----
#words=[word for word in tokens if word.isalpha()]

#----filter out stop words----
#stop_words = set(stopwords.words('english'))
#words = [w for w in words if not w in stop_words]
#print(words[:100])
#print(len(words))

#----STEMMING OF WORDS----
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])
