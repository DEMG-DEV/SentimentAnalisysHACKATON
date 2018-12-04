# SentimentAnalisysHACKATON
### About Practice Problem : Twitter Sentiment Analysis
**Sentiment Analysis** refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. *(Source: Wikipedia)*

### Data Science Resources
- Refer this comprehensive guide that exhaustively covers multiple techniques including TF-IDF, Word2Vec etc. to tackle this problem
- Are you a beginner? If yes, you can check out our latest 'Intro to Data Science' course to kickstart your journey in data science.

### Rules
- One person cannot participate with more than one user accounts.
- You are free to use any tool and machine you have rightful access to.
- You can use any programming language or statistical software.
- You are free to use solution checker as many times as you want.

### FAQs
**1. Are there any prizes/AV Points for this contest?**

This contest is purely for learning and practicing purpose and hence no participant is eligible for prize or AV points.

**2. Can I share my approach/code?**

Absolutely. You are encouraged to share your approach and code file with the community. There is even a facility at the leaderboard to share the link to your code/solution description.

**3. I am facing a technical issue with the platform/have a doubt regarding the problem statement. Where can I get support?**

Post your query on discussion forum at the thread for this problem, discussion threads are given at the bottom of this page. You could also join the AV slack channel by clicking on 'Join Slack Live Chat' button and ask your query at channel: practice_problems.

### Registration Fee
Free

___

### Problem Statement
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

### Motivation
Hate  speech  is  an  unfortunately  common  occurrence  on  the  Internet.  Often social media sites like Facebook and Twitter face the problem of identifying and censoring  problematic  posts  while weighing the right to freedom of speech. The  importance  of  detecting  and  moderating hate  speech  is  evident  from  the  strong  connection between hate speech and actual hate crimes. Early identification of users promoting  hate  speech  could  enable  outreach  programs that attempt to prevent an escalation from speech to action. Sites such as Twitter and Facebook have been seeking  to  actively  combat  hate  speech. In spite of these reasons, NLP research on hate speech has been very limited, primarily due to the lack of a general definition of hate speech, an analysis of its demographic influences, and an investigation of the most effective features.

___

### Data
Our overall collection of tweets was split in the ratio of 65:35 into training and testing data. Out of the testing data, 30% is public and the rest is private.

### Data Files
1. **train.csv** - For training the models, we provide a labelled dataset of 31,962 tweets. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet.
*There is 1 test file (public)*
2. **test_tweets.csv** - The test data file contains only tweet ids and the tweet text with each tweet in a new line.
 
### Submission Details
The following 3 files are to be uploaded.
1. **test_predictions.csv** - This should contain the 0/1 label for the tweets in test_tweets.csv, in the same order corresponding to the tweets in test_tweets.csv. Each 0/1 label should be in a new line.
2. **A .zip file of source code** - The code should produce the output file submitted and must be properly commented.
 

### Evaluation Metric:
The metric used for evaluating the performance of classification model would be F1-Score.

The metric can be understood as -

**True Positives (TP)** - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes.

**True Negatives (TN)** - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no.

**False Positives (FP)** – When actual class is no and predicted class is yes.

**False Negatives (FN)** – When actual class is yes but predicted class in no.

**Precision** = TP/TP+FP

**Recall** = TP/TP+FN

**F1 Score** = 2*(Recall * Precision) / (Recall + Precision)

F1 is usually more useful than accuracy, especially if for an uneven class distribution.

[Test File](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/download/test-file)

[Train File](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/download/train-file)

[Sample Submissions](https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/download/sample-submission)
