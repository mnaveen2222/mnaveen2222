# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 13:56:38 2018

@author: bharat
"""

import os


os.chdir('C:\\Users\\bharat\\Downloads')



import pandas as pd

data=pd.read_csv('train_E6oV3lV.csv')

data.columns

from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import pandas as pd 
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


tk=TweetTokenizer()
ps = PorterStemmer()
lem=WordNetLemmatizer()
def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s=s.lower()
    s=tk.tokenize(s)
    s=[ps.stem(word) for word in s if not word in set(stopwords.words('english'))]
    s=[lem.lemmatize(word) for word in s]
    s= ' '.join(s)
    return s




data['content'] = [cleaning(s) for s in data['tweet']]


all_words = ' '.join([text for text in data['content']])


data['content']





from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


normal_words =' '.join([text for text in data['content'][data['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



negative_words = ' '.join([text for text in data['content'][data['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht =re.findall(r'\w+', i)
        hashtags.append(ht)
    return hashtags


HT_regular = hashtag_extract(data['content'][data['label'] == 0])

HT_negative = hashtag_extract(data['content'][data['label'] == 1])



HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


import seaborn as sns

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()





b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
















from sklearn.feature_extraction.text import  CountVectorizer


vectorizer = CountVectorizer(max_features=3000,stop_words=stopwords.words('english')).fit(data['content'])



X=vectorizer.transform(data['content']).toarray()

from sklearn.feature_extraction.text import TfidfTransformer

tran=TfidfTransformer().fit(X)

X=tran.transform(X).toarray()


from sklearn.model_selection import train_test_split


train_x,test_x,train_y,text_y=train_test_split(X,data.label.values,test_size=0.25,random_state=0)


from sklearn.linear_model import LogisticRegression


model_log=LogisticRegression()


model_log.fit(train_x,train_y)


model_log.score(test_x,text_y)


model_log.score(X,data.label.values)



import xgboost


model_log=xgboost.XGBClassifier()




model_log.fit(train_x,train_y)


model_log.score(test_x,text_y)


model_log.score(X,data.label.values)


from sklearn.naive_bayes import GaussianNB



model_log.fit(X,data.label.values)


model_log.score(test_x,text_y)


model_log.score(X,data.label.values)

from sklearn.ensemble import RandomForestClassifier
model_log=RandomForestClassifier(n_estimators=100)


model_log.fit(X,data.label.values)


model_log.score(test_x,text_y)


model_log.score(X,data.label.values)




data['Predict']=model_log.predict(X)











from sklearn.model_selection import cross_val_score,KFold



kfold=KFold(n_splits=10)


score=cross_val_score(model,X,data.label.values,cv=kfold,scoring="accuracy")
score.mean()
score


data_test=pd.read_csv('test_tweets_anuFYb8.csv')





data_test['content'] = [cleaning(s) for s in data_test['tweet']]


X_test=vectorizer.transform(data_test['content']).toarray()



X_test=tran.fit_transform(X_test).toarray()





data_test['Predict']=model_log.predict(X_test)

data_test.columns=['id', 'label']

data_test.drop(['tweet','content'],axis=1,inplace=True)


data_test.columns


data_test.columns=['id', 'label']






data_test.to_csv('SampleSubmission2.csv',index = False)












