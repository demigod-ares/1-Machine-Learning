# Tokenization & Text Normalization using NLTK

# tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
text = "Hi, John! How are you? Hope you are donig great. I will be visiting your city this weekend. Let's catchup!"
sentences = sent_tokenize(text)
words = word_tokenize(text)
# Ngrams tokenization
from nltk import ngrams
for gram in ngrams(word_tokenize(text), 2):
    print(gram)

# Normaliztion (Stemming)
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem("plays"))
print(stemmer.stem("playing"))
print(stemmer.stem("played"))
print(stemmer.stem("fucks"))
print(stemmer.stem("fucking"))
print(stemmer.stem("fucked"))
# Bad result
print(stemmer.stem("increases"))

# Better Noramlisation (using Lemmatization)
from nltk.stem import WordNetLemmatizer
lemm = WordNetLemmatizer()
print(lemm.lemmatize("increases"))
print(lemm.lemmatize("running"))
# using parts of speech tag now
print(lemm.lemmatize("running", pos="v"))

# get the parts of speech tag
from nltk import pos_tag
pos_tag(words)

from nltk.corpus import wordnet
wordnet.synsets("good")