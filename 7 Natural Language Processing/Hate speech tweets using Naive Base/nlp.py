# 1. Load the dataset
import pandas as pd
dataset = pd.read_csv('hate_speech_tweets.csv')
'''label is the column that contains the target variable or the value that has to be predicted.
1 means it's a hate speech and 0 means it is not.'''
# Check Noise present in Tweets
for index, tweet in enumerate(dataset["tweet"][10:15]):
    print(index+1,".",tweet)
'''If you look closely, you'll see that there are many hashtags present in the tweets of the form # symbol followed by text.
We particularly don't need the # symbol so we will clean it out.
Also, there are strange symbols like â and ð in tweet 4.
This is actually unicode characters that is present in our dataset that we need to get rid of because they don't particularly add anything meaningful.
There are also numerals and percentages.'''

# 2. clean up the noise in our dataset
import re
# creating cleaning function
def clean_text(text):
    #Filter to allow only alphabets
    text = re.sub(r'[^a-zA-Z\']', ' ', text)    
    #Remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    #Convert to lowercase to maintain consistency
    text = text.lower()       
    return text
# Clean text from noise
dataset['clean_text'] = dataset.tweet.apply(lambda x: clean_text(x))

'''3. Feature Engineering
Feature engineering is the science (and art) of extracting more information from existing data.
You are not adding any new data here, but you are actually making the data you already have more useful.
The machine learning model does not understand text directly, so we create numerical features that reperesant the underlying text.
In this module, you'll deal with very basic NLP based features and as you progress further in the course you'll come across more complex and efficient ways of doing the same.'''

#Exhaustive list of stopwords in the english language. We want to focus less on these so at some point will have to filter
STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'also', 'am', 'an', 'and',
              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'com', 'could', "couldn't", 'did',
              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'else', 'ever',
              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it',
              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself',
              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'otherwise', 'ought', 'our', 'ours',
              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "shan't", 'she', "she'd", "she'll", "she's",
              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs',
              'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're",
              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't",
              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
              "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't",
              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

# Generate word frequency
def gen_freq(text):
    #Will store the list of words
    word_list = []
    #Loop over all the tweets and extract words into word_list
    for tw_words in text.split():
        word_list.extend(tw_words)
    #Create word frequencies using word_list
    word_freq = pd.Series(word_list).value_counts()    
    #Drop the stopwords during the frequency calculation
    word_freq = word_freq.drop(STOP_WORDS, errors='ignore')    
    return word_freq

#Check whether a negation term is present in the text
def any_neg(words):
    for word in words:
        if word in ['n', 'no', 'non', 'not'] or re.search(r"\wn't", word):
            return 1
    else:
        return 0

#Check whether one of the 100 rare words is present in the text
def any_rare(words, rare_100):
    for word in words:
        if word in rare_100:
            return 1
    else:
        return 0

#Check whether prompt words are present
def is_question(words):
    for word in words:
        if word in ['when', 'what', 'how', 'why', 'who']:
            return 1
    else:
        return 0

word_freq = gen_freq(dataset.clean_text.str)
#Number of words in a tweet
dataset['word_count'] = dataset.clean_text.str.split().apply(lambda x: len(x))
#Negation present or not
dataset['any_neg'] = dataset.clean_text.str.split().apply(lambda x: any_neg(x))
#Prompt present or not
dataset['is_question'] = dataset.clean_text.str.split().apply(lambda x: is_question(x))
#100 most rare words in the dataset
rare_100 = word_freq[-100:]
#Any of the most 100 rare words present or not
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda x: any_rare(x, rare_100))
#Character count of the tweet
dataset['char_count'] = dataset.clean_text.apply(lambda x: len(x))
#Top 10 common words are
gen_freq(dataset.clean_text.str)[:10]

# Splitting data into training set and test set.
from sklearn.model_selection import train_test_split
X = dataset[['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']]
y = dataset.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27)

'''4. Train an ML model for Text Classification
Now that the dataset is ready, it is time to train a Machine Learning model on the same.
You will be using a Naive Bayes classifier from sklearn which is a prominent python library used for machine learning.'''

from sklearn.naive_bayes import GaussianNB
#Initialize GaussianNB classifier
model = GaussianNB()
#Fit the model on the train dataset
model = model.fit(X_train, y_train)
#Make predictions on the test dataset
y_pred = model.predict(X_test)

# 5. Evaluate the ML model
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")
cm = confusion_matrix(y_test, y_pred)

