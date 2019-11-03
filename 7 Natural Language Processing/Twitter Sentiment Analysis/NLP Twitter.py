# Twitter sentiment analysis
import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
train  = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv')
# Data Inspection. Let’s check out a few non racist/sexist tweets.
train[train['label'] == 0].head(10)
# Now check out a few racist/sexist tweets.
train[train['label'] == 1].head(10)
# Let’s check dimensions of the train and test dataset.
train.shape, test.shape
# Let’s have a glimpse at label-distribution in the train dataset.
train["label"].value_counts()
# Now we will check the distribution of length of the tweets in terms of words, in both train and test data.
length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len() 
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.xlabel("No. of words in tweets")
plt.ylabel("No. of users")
plt.legend() 
plt.show()
# Before we begin cleaning, let’s first combine train and test datasets.
# Combining the datasets will make it convenient for us to preprocess the data.
# Later we will split it back into train and test data.
combi = train.append(test, ignore_index=True) 
combi.shape
# Given below is a user-defined function to remove unwanted text patterns from the tweets.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt
# 1. Removing Twitter Handles (@user)
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 
# 2. Removing Punctuations, Numbers, and Special Characters
combi['tidy_tweet'] = combi['tweet'].str.replace("[^a-zA-Z#]", " ")     
# 3. Removing Short Words
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
# 4. Text Normalization
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 
# Now we can normalize the tokenized tweets.
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
#Now let’s stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet

# A) Understanding the common words used in the tweets: WordCloud
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
# B) Words in non racist/sexist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
# C) Racist/Sexist Tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
''' D) Understanding the impact of Hashtags on tweets sentiment
We will store all the trend terms in two separate lists — one for non-racist/sexist tweets and the other for racist/sexist tweets.'''
# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)        
        hashtags.append(ht)
    return hashtags
# extracting hashtags from non racist/sexist tweets 
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) 
# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1]) 
# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

# Non-Racist/Sexist Tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 
# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
# Racist/Sexist Tweets
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())}) 
# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n = 20)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

'''To analyse a preprocessed data, it needs to be converted into features.
Depending upon the usage, text features can be constructed using assorted techniques:
1. Bag of Words 2. TF-IDF 3. Word Embeddings. Read on to understand these techniques in detail.'''
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape
# TF-IDF Features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape
# 1. Word2Vec Embedding (Word to Vector)
'''We will train a Word2Vec model on our data to obtain vector representations for all the unique words present in our corpus.
Word2Vec is not a single algorithm but a combination of two techniques:
    1. CBOW (Continuous bag of words) and 2. Skip-gram model.
Both of these are shallow neural networks which map word(s) to the target variable which is also a word(s).
Both of these techniques learn weights which act as word vector representations.
1. CBOW tends to predict the probability of a word given a context.
A context may be a single adjacent word or a group of surrounding words.
2. The Skip-gram model works in the reverse manner.
It tries to predict the context for a given word.
We will go ahead with the Skip-gram model as it has the following advantages:
1. It can capture two semantics for a single word. i.e it will have two vector representations of ‘apple’.
One for the company Apple and the other for the fruit.
2. Skip-gram with negative sub-sampling outperforms CBOW generally.
There is one more option of using pre-trained word vectors instead of training our own model.
Some of the freely available pre-trained vectors are:
1. Google News Word Vectors
2. Freebase names
3. DBPedia vectors (wiki2vec)
However, for this course, we will train our own word vectors.
Since size of the pre-trained word vectors is generally huge.
Let’s train a Word2Vec model on our corpus.'''
import gensim
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing 
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet, size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2, sg = 1, # 1 for skip-gram model
            hs = 0, negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34) 
model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)
'''Let’s play a bit with our Word2Vec model and see how does it perform.
We will specify a word and the model will pull out the most similar words from the corpus.'''
model_w2v.wv.most_similar(positive="dinner")
model_w2v.wv.most_similar(positive="trump")
# Let’s check the vector representation of any word from our corpus.
model_w2v['food']
len(model_w2v['food']) #The length of the vector is 200
# We will use the below function to create a vector for each tweet by taking the average of the vectors of the words present in the tweet.
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec
# Preparing word2vec feature set…
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_df = pd.DataFrame(wordvec_arrays)
wordvec_df.shape

# 2. Doc2Vec Embedding
'''Doc2Vec model is an unsupervised algorithm to generate vectors for sentence/paragraphs/documents.
This approach is an extension of the word2vec. The major difference between the two is that doc2vec provides an additional context which is unique for every document in the corpus.
This additional context is nothing but another feature vector for the whole document.
This document vector is trained along with the word vectors.'''
# Let’s load the required libraries.
from tqdm import tqdm
tqdm.pandas(desc="progress-bar") 
from gensim.models.doc2vec import LabeledSentence
# To implement doc2vec, we have to labelise or tag each tokenised tweet with unique IDs.
# We can do so by using Gensim’s LabeledSentence() function.
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet) # label all the tweets
# Let’s have a look at the result. Check labeled_tweets variable
labeled_tweets[:6]

# Now let’s train a doc2vec model.
model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model                                   
                                  dm_mean=1, # dm = 1 for using mean of the context word vectors                                  
                                  size=200, # no. of desired features                                  
                                  window=5, # width of the context window                                  
                                  negative=7, # if > 0 then negative sampling will be used
                                  min_count=5, # Ignores all words with total frequency lower than 2.                                  
                                  workers=3, # no. of cores                                  
                                  alpha=0.1, # learning rate                                  
                                  seed = 23) 
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)

# Preparing doc2vec Feature Set
docvec_arrays = np.zeros((len(tokenized_tweet), 200))
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1, 200))
docvec_df = pd.DataFrame(docvec_arrays) 
docvec_df.shape

'''We will use the following algorithms to build models:
1.Logistic Regression
2.Support Vector Machine
3.RandomForest
4.XGBoost

F1 score is being used as the evaluation metric.
It is the weighted average of Precision and Recall.
Therefore, this score takes both false positives and false negatives into account.
It is suitable for uneven class distribution problems.
The important components of F1 score are:
1.True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.
2.True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.
3.False Positives (FP) – When actual class is no and predicted class is yes.
4.False Negatives (FN) – When actual class is yes but predicted class in no.
5.Precision = TP/TP+FP
6.Recall = TP/TP+FN
7.F1 Score = 2(Recall Precision) / (Recall + Precision)'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
# We will first try to fit the logistic regression model on the Bag-of_Words (BoW) features
# Extracting train and test BoW features
train_bow = bow[:31962,:]
test_bow = bow[31962:,:] 
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],
                                                          random_state=42,
                                                          test_size=0.3)
lreg = LogisticRegression() 
# training the model
lreg.fit(xtrain_bow, ytrain) 
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # calculating f1 score = 0.531
# Now let’s make predictions for the test dataset
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
# TF-IDF Features fitted using logistic regression
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:] 
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
lreg.fit(xtrain_tfidf, ytrain) 
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # calculating f1 score for the validation set ie. 0.544
# Word2Vec Features  fitted using logistic regression
train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:] 
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
lreg.fit(xtrain_w2v, ytrain) 
prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # score = 0.622
# Doc2Vec Features fitted using logistic regression
train_d2v = docvec_df.iloc[:31962,:]
test_d2v = docvec_df.iloc[31962:,:] 
xtrain_d2v = train_d2v.iloc[ytrain.index,:]
xvalid_d2v = train_d2v.iloc[yvalid.index,:]
lreg.fit(xtrain_d2v, ytrain) 
prediction = lreg.predict_proba(xvalid_d2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # score = 0.367

from sklearn import svm
# Support Vector Machine (SVM) on the Bag-of_Words (BoW) features
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain) 
prediction = svc.predict_proba(xvalid_bow) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # score = 0.508
test_pred = svc.predict_proba(test_bow) 
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
# TF-IDF Features fitted using Support Vector Machine (SVM)
svc = svm.SVC(kernel='linear', 
C=1, probability=True).fit(xtrain_tfidf, ytrain) 
prediction = svc.predict_proba(xvalid_tfidf) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # score = 0.51
# Word2Vec Features  fitted using Support Vector Machine (SVM)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain) 
prediction = svc.predict_proba(xvalid_w2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # score = 0.614
# Doc2Vec Features fitted using Support Vector Machine (SVM)
svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_d2v, ytrain) 
prediction = svc.predict_proba(xvalid_d2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # score = 0.203

from sklearn.ensemble import RandomForestClassifier
# Random Forest on the Bag-of_Words (BoW) features
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain) 
prediction = rf.predict(xvalid_bow) 
f1_score(yvalid, prediction) # score = 0.553
test_pred = rf.predict(test_bow)
# TF-IDF Features fitted using Random Forest
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain) 
prediction = rf.predict(xvalid_tfidf)
f1_score(yvalid, prediction) # score = 0.562
# Word2Vec Features  fitted using Random Forest
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain) 
prediction = rf.predict(xvalid_w2v)
f1_score(yvalid, prediction) # score = 0.507
# Doc2Vec Features fitted using Random Forest
rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain) 
prediction = rf.predict(xvalid_d2v)
f1_score(yvalid, prediction) #score = 0.056

'''Extreme Gradient Boosting (xgboost) is an advanced implementation of gradient boosting algorithm.
It has both linear model solver and tree learning algorithms.
Its ability to do parallel computation on a single machine makes it extremely fast.
It also has additional features for doing cross validation and finding important variables.
There are many parameters which need to be controlled to optimize the model.
Some key benefits of XGBoost are:
1.Regularization - helps in reducing overfitting
2.Parallel Processing - XGBoost implements parallel processing and is blazingly faster as compared to GBM.
3.Handling Missing Values - It has an in-built routine to handle missing values.
4.Built-in Cross-Validation - allows user to run a cross-validation at each iteration of the boosting process'''
from xgboost import XGBClassifier
# Extreme Gradient Boosting (xgboost) on the Bag-of_Words (BoW) features
xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)
prediction = xgb_model.predict(xvalid_bow)
f1_score(yvalid, prediction) # score = 0.513
test_pred = xgb_model.predict(test_bow)
# TF-IDF Features fitted using Extreme Gradient Boosting (xgboost)
xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain) 
prediction = xgb.predict(xvalid_tfidf)
f1_score(yvalid, prediction) # score = 0.503
# Word2Vec Features  fitted using Extreme Gradient Boosting (xgboost)
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain) 
prediction = xgb.predict(xvalid_w2v)
f1_score(yvalid, prediction) # score = 0.652
# Doc2Vec Features fitted using Extreme Gradient Boosting (xgboost)
xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain) 
prediction = xgb.predict(xvalid_d2v)
f1_score(yvalid, prediction) # score = 0.345
# Word2Vec Features modelling with xgboost WINS!!!

# Parameter tuning using XG Boost
import xgboost as xgb
# Here we will use DMatrices. A DMatrix can contain both the features and the target.
dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain) 
dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid) 
dtest = xgb.DMatrix(test_w2v)
# Parameters that we are going to tune 
params = {
            'objective':'binary:logistic',
            'max_depth':6,
            'min_child_weight': 1,
            'eta':.3,
            'subsample': 1,
            'colsample_bytree': 1
         }
# We will prepare a custom evaluation metric to calculate F1 score.
def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]
# Tuning max_depth and min_child_weight
gridsearch_params = [
                        (max_depth, min_child_weight)
                        for max_depth in range(7,10)
                            for min_child_weight in range(5,8)
                    ]
max_f1 = 0. # initializing with 0 
best_params = None 
for max_depth, min_child_weight in gridsearch_params:
    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
     # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

     # Cross-validation
    cv_results = xgb.cv(        
                        params, dtrain, feval= custom_eval,
                        num_boost_round=200, maximize=True,
                        seed=16, nfold=5, early_stopping_rounds=10
                        )     
    # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()    
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))    
    if mean_f1 > max_f1:
            max_f1 = mean_f1
            best_params = (max_depth,min_child_weight) 
    print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
# Best params: 8, 5, F1 Score: 0.6822174000000001
# mine: Best params: 8, 5, F1 Score: 0.6762248
params['max_depth'] = 8 
params['min_child_weight'] = 6

# Tuning subsample and colsample
gridsearch_params = [
    (subsample, colsample)
    for subsample in [i/10. for i in range(8,11)]
    for colsample in [i/10. for i in range(5,8)] ]
max_f1 = 0. 
best_params = None 
for subsample, colsample in gridsearch_params:
    print("CV with subsample={}, colsample={}".format(
                             subsample,
                             colsample))
     # Update our parameters
    params['colsample'] = colsample
    params['subsample'] = subsample
    cv_results = xgb.cv(
                        params,
                        dtrain,
                        feval= custom_eval,
                        num_boost_round=200,
                        maximize=True,
                        seed=16,
                        nfold=5,
                        early_stopping_rounds=10
                        )
     # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (subsample, colsample)
print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))
# params: 0.9, 0.5, F1 Score: 0.6830936000000001
params['subsample'] = .9 # mine = 1.0
params['colsample_bytree'] = .5

# Now let’s tune the learning rate.
max_f1 = 0. 
best_params = None 
for eta in [.2, .1, .05]:
    print("CV with eta={}".format(eta))
    # Update ETA
    params['eta'] = eta
    # Run Cross-Validation
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=1000,
        maximize=True,
        seed=16,
        nfold=5,
        early_stopping_rounds=20
    )
    # Finding best F1 Score
    mean_f1 = cv_results['test-f1_score-mean'].max()
    boost_rounds = cv_results['test-f1_score-mean'].argmax()
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = eta 
print("Best params: {}, F1 Score: {}".format(best_params, max_f1))
# Best params: 0.1, F1 Score: 0.6864700000000001
params['eta'] = .1

# finally
xgb_model = xgb.train(
    params,
    dtrain,
    feval= custom_eval,
    num_boost_round= 1000,
    maximize=True,
    evals=[(dvalid, "Validation")],
    early_stopping_rounds=10
 )

test_pred = xgb_model.predict(dtest)
test['label'] = (test_pred >= 0.3).astype(np.int)

'''WHAT ELSE CAN BE TRIED?
We have covered a lot in this Sentiment Analysis course, but still there is plenty of room for other things to try out. Given below is a list of tasks that you can try with this data.

We have built so many models in this course, we can definitely try model ensembling. A simple ensemble of all the submission files (maximum voting) yielded an F1 score of 0.55 on the public leaderboard.

1.Use Parts-of-Speech tagging to create new features.
2.Use stemming and/or lemmatization. It might help in getting rid of unnecessary words.
3.Use bi-grams or tri-grams (tokens of 2 or 3 words respectively) for Bag-of-Words and TF-IDF.
4.We can give pretrained word-embeddings models a try.'''
