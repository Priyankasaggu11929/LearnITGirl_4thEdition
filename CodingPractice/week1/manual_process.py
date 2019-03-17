#import re
import string

#-----load data-----
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()

#-----split into words by white space----
words = text.split()
#words = re.split(r'\W+', text)

#-----covert to lower case----
words =[word.lower() for word in words]
#words=[word.upper() for word in words]
print(words[:100])

#----remove punctuation from each word---
#table = str.maketrans('', '', string.punctuation)
#print(table)
#stripped = [w.translate(table) for w in words]
#print(stripped[:100])


