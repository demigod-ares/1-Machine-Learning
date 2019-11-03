# Tokenisation (Corpus, Tokens and Ngrams) & Text Normalisation (stemming and lemmatization)

""" Why to Preprocess text data?
As you may have already seen that without performing preprocessing operations like cleaning, removing stopwords and changing case in the dataset the representation always comes out wrong.
In this case, it was that the wordcloud was full of noise but in other cases it might be your Machine Learning model that is going to suffer."""
#Load the dataset
import pandas as pd 
dataset = pd.read_csv('tweets.csv', encoding = 'ISO-8859-1')

#  Generating Word Frequency
def gen_freq(text):
    #Will store the list of words
    word_list = []
    #Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)
    #Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()    
    return word_freq
# Generate word frequencies
word_freq = gen_freq(dataset.text.str)

# EDA(Exploratory data analysis) using Word Clouds
# Import libraries
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Generate word cloud
wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)
# Plot the wordcloud
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

'''Few things to Note:-

There is noise in the form of "RT" and "&amp" which can be removed from the word frequency.
1.Stop words like "the", "in", "to", "of" etc. are obviously ranking among the top frequency words but these are just constructs of the English language and are not specific to the people's tweets.
2.Words like "demonetization" have occured multiple times. The reason for this is that the current text is not Normalized so words like "demonetization", "Demonetization" etc. are all considered as different words.
3.The above are some of the problems that we need to address in order to make better visualization. Let's solve some of the problems!'''

# utilize Regex to do text cleaning
import re

def clean_text(text):
    #Remove RT
    text = re.sub(r'RT', '', text)    
    #Fix &
    text = re.sub(r'&amp;', '&', text)    
    #Remove punctuations
    text = re.sub(r'[?!.;:,#@-]', '', text)
    #Convert to lowercase to maintain consistency
    text = text.lower()
    return text

# Import list of stopwards from wordcloud
from wordcloud import STOPWORDS
print(STOPWORDS)
# Stopwords Removal
text = dataset.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')

# Generate word cloud again
wc = WordCloud(width=450, height=330, max_words=200, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

'''Now that you have succesfully created a wordcloud, you can get some insight into the areas of interest of the general twitter users:

It is evident that people are talking about govt. policies like demonetization, J&K.
There are some personalitites that are mentioned numerous times like evanspiegel, PM Narendra Modi, Dr Kumar Vishwas etc.
There are also talks about oscars, youtube and terrorists
There are many sub-topics that revolve around demonetization like atms, bank, cash, paytm etc. Which tells that many people are concerned about it.

Challenge!!!
As you would have noticed, even the current word cloud has some form of noise especially from strange symbols like <U...>.
It's your task to go ahead and figure out how to deal with the, given the fact that they are present in word cloud implies that noise is widely present in the data.

Also something to note is even now some words are misreperesented for example: modi, narendra and narendramodi all refer to the same person.
This can eaisly be solved by Normalizing our text which is a technique that you'll learn in future models.'''