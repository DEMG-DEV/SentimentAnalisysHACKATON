import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('Data/train_E6oV3lV.csv')
test = pd.read_csv('Data/test_tweets_anuFYb8.csv')

combined = train.append(test, ignore_index=True)


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


# remove twitter handles (@user)
combined['tidy_tweet'] = np.vectorize(
    remove_pattern)(combined['tweet'], "@[\w]*")

# remove special characters, numbers, punctuations
combined['tidy_tweet'] = combined['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")

"""# remove all the words with 3 of lenght
combined['tidy_tweet'] = combined['tidy_tweet'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))"""

tokenized_tweet = combined['tidy_tweet'].apply(lambda x: x.split())

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(
    lambda x: [stemmer.stem(i) for i in x])  # stemming

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combined['tidy_tweet'] = tokenized_tweet


def hashtag_extract(x):
    """ function to collect hashtag"""
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(
    max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combined['tidy_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combined['tidy_tweet'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962, :]
test_bow = bow[31962:, :]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(
    train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain)  # training the model

prediction = lreg.predict_proba(xvalid_bow)  # predicting on the validation set
# if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction[:, 1] >= 0.3
prediction_int = prediction_int.astype(np.int)

print("F1 Score")
print(f1_score(yvalid, prediction_int))  # calculating f1 score

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:, 1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id', 'label']]
# writing data to a CSV file
submission.to_csv('sub_lreg_bow.csv', index=False)
