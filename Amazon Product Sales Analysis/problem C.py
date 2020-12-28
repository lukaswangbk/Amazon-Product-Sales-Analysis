# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:09:06 2020

@author: lukas
"""

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import chart_studio.plotly as plty
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import os

# Importing the dataset
df_hair_dryer = pd.read_csv("hair_dryer.tsv", sep="\t", quoting = 3)
df_microwave = pd.read_csv("microwave.tsv", sep="\t", quoting = 3)
df_pacifier = pd.read_csv("pacifier.tsv", sep="\t", quoting = 3)

'''
---------------------------------1----------------------------------
'''

# Adding a length column for analyzing the length of the reviews
df_hair_dryer['length'] = df_hair_dryer['review_body'].apply(len)
#df_hair_dryer.groupby('length').describe().sample(10)
#df_hair_dryer.groupby('star_rating').describe()

## Adding a feedback column for analyzing the success based on star_rating
#df_hair_dryer['feedback'] = 0
#df_hair_dryer.loc[df_hair_dryer.star_rating>3, 'feedback'] = 2
#df_hair_dryer.loc[df_hair_dryer.star_rating==3, 'feedback'] = 1
#df_hair_dryer.loc[df_hair_dryer.star_rating<3, 'feedback'] = 0
#df_hair_dryer.groupby('feedback').describe()

# Visualization for distribution of star_rating for hair dryer
star_rating = df_hair_dryer['star_rating'].value_counts()
label_rating = star_rating.index
size_rating = star_rating.values
colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']
rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'hair_dryer', hole = 0.3)
df = [rating_piechart]
layout = go.Layout(title = 'Distribution of star_rating for hair dryer')
fig = go.Figure(data = df, layout = layout)
py.plot(fig)

# 3D-Visualization for realation of (1)length (2)star rating (3)product id
trace = go.Scatter3d(
    x = df_hair_dryer['length'],
    y = df_hair_dryer['star_rating'],
    z = df_hair_dryer['product_id'],
    name = 'hair dryer',
    mode='markers',
    marker=dict(
        size=5,
        color = df_hair_dryer['star_rating'],
        colorscale = 'Viridis',
    )  
)
df = [trace]

layout = go.Layout(
    title = 'Length vs Product_ID vs Star_rating',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
)
fig = go.Figure(data = df, layout = layout)
py.plot(fig)

# Visualization for the reviews length
color = 'tomato'
df_hair_dryer['length'].value_counts().plot.hist(color = color, figsize = (20, 10), bins = 50)
plt.title('Distribution of Review Length in Reviews', fontsize = 50)
plt.xlabel('review lengths', fontsize = 20)
plt.ylabel('count', fontsize = 20)
plt.xticks(np.arange(0, 100, step=5))
plt.show()

# Visualization for sales of different product
color = plt.cm.copper(np.linspace(0, 1, 50))
df_hair_dryer['product_id'].value_counts().nlargest(50).plot.bar(color = color, figsize = (20, 10))
plt.title('Distribution of Sales of Different Product (Top 50)', fontsize = 50)
plt.xlabel('product_id', fontsize = 20)
plt.ylabel('count', fontsize = 20)
plt.show()

# Visualization for star_rating of review length
# Boxplot
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')
sns.boxplot(df_hair_dryer['star_rating'], df_hair_dryer['length'], palette = 'Blues')
plt.title("Relations between Review Length and Star Rating", fontsize = 50)
plt.show()

# Stripplot
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')
sns.stripplot(df_hair_dryer['star_rating'], df_hair_dryer['length'], palette = 'Reds')
plt.title("Relations between Review Length and Star Rating", fontsize = 50)
plt.show()

'''
---------------------------------2----------------------------------
'''

'''
------------------------------Part a--------------------------------
'''

# Cleaning the reviews
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, df_hair_dryer.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df_hair_dryer['review_body'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
words = cv.fit_transform(corpus)
feature_name = cv.get_feature_names()

# Finding most frequently words
sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

# Visualize the most frequently words
# Bar
plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 50))
frequency.head(50).plot(x='word', y='freq', kind='bar', figsize=(20, 10), fontsize = 20, color = color)
plt.title("Most Frequently Occuring Words (Top 50)", fontsize = 50)
plt.show()

# Words clouds
from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000).generate_from_frequencies(dict(words_freq[:500]))
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Vocabulary from Reviews (Top 500)", fontsize = 50)
plt.show()


'''
------------------------------Part b--------------------------------
'''

# Time-based (year) reputation
# Extracting year from review_date
df_hair_dryer["review_date"] = pd.to_datetime(df_hair_dryer["review_date"], format='%m/%d/%Y')
df_hair_dryer['year'], df_hair_dryer['month']= df_hair_dryer['review_date'].dt.year, df_hair_dryer['review_date'].dt.month

# Calculating avg_star_rating and total_comments
avg_star_rating = []
total_comments  = []
for i in range(min(df_hair_dryer['year']), max(df_hair_dryer['year'])+1):
    avg_star_rating.append((df_hair_dryer[df_hair_dryer['year'] == i]["star_rating"].sum())/(df_hair_dryer[df_hair_dryer['year'] == i].shape[0]))
    total_comments.append((df_hair_dryer[df_hair_dryer['year'] == i].shape[0]))

# Find avg_star_rating and total_comments for each year respectively
tmp={"avg_star_rating" : avg_star_rating,
     "total_comments" : total_comments}
tmp_df=DataFrame(tmp)
tmp_df["year"]=[2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]

# Visualize average star rating and comments number for each year
sns.set_style("whitegrid")
 
x  = tmp_df["year"]
y1 = tmp_df['total_comments']
y2 = tmp_df['avg_star_rating']

plt.rcParams['figure.figsize'] = (20,10)
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.bar(x, y1, alpha=.7, color='g')
ax1.set_ylabel('number_of_comments',fontsize='20')
ax1.set_title("Average Star Rating and Comments Number in 2002-2015",fontsize='50')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'orange', ms=10, marker='*', alpha=0.5, linewidth=5)
ax2.set_ylabel('average_star_rating',fontsize='20')

'''
------------------------------Part c--------------------------------
'''

# Generate dataset with sales volumn, average star rating and reviews based on product ID
tmp_df1 = df_hair_dryer.loc[:,'product_id'].value_counts()
grouped=df_hair_dryer['star_rating'].groupby(df_hair_dryer['product_id'])
tmp_df2 = grouped.mean()

tmp_df = pd.merge(tmp_df1, tmp_df2, how='left', left_index=True, right_index=True)
tmp_df.rename(columns={"product_id": "sales", "star_rating": "avg_star_rating"}, inplace=True)

tmp_df3 = DataFrame(cv.fit_transform(corpus).toarray())
tmp_df3["product_id"] = df_hair_dryer["product_id"]
tmp_df3 = tmp_df3.groupby(by=["product_id"]).sum()
tmp_df = pd.merge(tmp_df, tmp_df3, how='left', left_index=True, right_index=True)

# Calibrating label for "success"
# tmp_df["sales"].describe()
tmp_df["success"] = 0
tmp_df.loc[tmp_df.sales>=10,'success'] = 1
tmp_df.loc[tmp_df.sales <10,'success'] = 0

# Build a model to find the most important features for success
# Load the data
X = tmp_df.iloc[:, 1:-1].values
y = tmp_df.iloc[:, -1].values
word = cv.fit_transform(corpus).toarray()
word_df = pd.DataFrame(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# define a dictionary for different classifiers and their parameters
classifiers = {
    "Dummy"        : DummyClassifier(strategy='uniform', random_state=2),
    "KNN(3)"       : KNeighborsClassifier(3), 
    "RBF SVM"      : SVC(gamma=2, C=1), 
    "Decision Tree": DecisionTreeClassifier(max_depth=7), 
    "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4), 
    "xgboost"      : XGBClassifier(),
    "Neural Net"   : MLPClassifier(alpha=1), 
    "AdaBoost"     : AdaBoostClassifier(),
    "Naive Bayes"  : GaussianNB(), 
    "QDA"          : QuadraticDiscriminantAnalysis(),
    "Linear SVC"   : LinearSVC(),
    "Linear SVM"   : SVC(kernel="linear"), 
    "Gaussian Proc": GaussianProcessClassifier(1.0 * RBF(1.0)),
}
from time import time
nfast = 10      # Run the first nfast learner. Don't run the very slow ones at the end
head = list(classifiers.items())[:nfast]

for name, classifier in head:
    start = time()                     # remember starting training time
    classifier.fit(X_train, y_train)
    train_time = time() - start        # get the total training time
    start = time()
    score = classifier.score(X_test, y_test)
    score_time = time()-start         # get the score time
    print("{:<15}| score = {:.3f} | time = {:,.3f}s/{:,.3f}s".format(name, score, train_time, score_time))

# Fitting AdaBoost Classification to the Training set
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
start = time()  
classifier.fit(X_train, y_train)
train_time = time() - start 
start = time()  
score = classifier.score(X_test, y_test)
score_time = time() - start 
print("score = {:.3f} | time = {:,.3f}s/{:,.3f}s".format(score, train_time, score_time))

# Calculating feature inportance
feature_name = cv.get_feature_names()
feature_name = np.array(feature_name)
feature_name = np.insert(feature_name, 0, "avg_star_rating", axis=0)
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_name = np.array(feature_name)
for f in range(100):
    print("%2d) %-*s %f" % (f + 1, 30, feature_name[indices[f]], importances[indices[f]]))
# 1) dryer                          0.155159
# 2) good                           0.152893
# 3) perfect                        0.087186
# 4) make                           0.072355
# 5) enough                         0.048774
# 6) job                            0.046361
# 7) love                           0.044690
# 8) nice                           0.033707
# 9) set                            0.031481
#10) heat                           0.025067
#11) product                        0.024280
#12) easi                           0.019226
#13) much                           0.018719
#14) friend                         0.013313
#15) blow                           0.013256
#16) purchas                        0.012955
#17) need                           0.012720
#18) went                           0.011697
#19) fast                           0.010531
#20) great                          0.010311
    
# 1) disappoint                     0.000000
# 2) problema                       0.000000
# 4) probelm                        0.000000
# 5) prize                          0.000000
# 6) privat                         0.000000
# 7) pristin                        0.000000
# 8) prioriti                       0.000000
# 9) problemat                      0.000000
#10) prior                          0.000000
#11) principl                       0.000000
#12) princess                       0.000000
#13) pride                          0.000000
#14) pricey                         0.000000
#15) print                          0.000000
#16) prolong                        0.000000
#17) prohibit                       0.000000
#18) program                        0.000000
#19) profit                         0.000000
#20) profession                     0.000000

# Visualization
# Barplot
from pandas.core.frame import DataFrame
tmp={"feature_name" : feature_name,
     "importances" : importances}
tmp_df=DataFrame(tmp)
tmp_df.sort_values("importances",ascending=False,inplace=True)
plt.figure(figsize=(20, 10))
plt.title("Feature Importance Ranking (Top 50)", fontsize = 50)
plt.xticks(rotation=90)
sns.barplot(x=tmp_df["feature_name"][:50], y=tmp_df["importances"][:50], data=tmp_df, palette="Paired")  
   
# Words clouds
from wordcloud import WordCloud
pos_words="disappoint problema probelm prize privat pristin prioriti worri prior principl princess pride pricey bad print useless prolong prohibit program profit profession producto productif prod proclaim processor pricetag priceless predecessor precis prayer pray predomin prais practicl practic howev danger surfac lost anoth low pow powder previsto previous previou previo prevail preview"
wc = wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000)
wc.generate(pos_words)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Important Words for Failure (top 50)", fontsize = 50)
plt.show()

pos_words="dryer good perfect make enough job love nice set heat product easi much friend blow purchas need went fast great like hair one excel size power model salon happi wife soft save strong often qualiti satisfi highli time realli conair placement use want reliabl lot hour loud sturdi deduct help"
wc = wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000)
wc.generate(pos_words)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Important Words for Success (top 50)", fontsize = 50)
plt.show()

'''
------------------------------Part d--------------------------------
'''

# Visualize number of reviews changes related based on star rating from 2002-2015

data_df = df_hair_dryer[["year", "star_rating"]]
                         
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (30, 18)
plt.title("Number of Reviews with Different Star Rating in 2002-2015", fontsize = 50)
sns.countplot(y="year", hue="star_rating", data=data_df, palette="Set2")
plt.show()

'''
------------------------------Part e--------------------------------
'''

# Build a model to find the most important words for star rating
# Load the data
X = cv.fit_transform(corpus).toarray()
y = df_hair_dryer.loc[:, "star_rating"].values
word_df = pd.DataFrame(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model Selection
# define a dictionary for different classifiers and their parameters
classifiers = {
    "Dummy"        : DummyClassifier(strategy='uniform', random_state=2),
    "KNN(3)"       : KNeighborsClassifier(3), 
    "RBF SVM"      : SVC(gamma=2, C=1), 
    "Decision Tree": DecisionTreeClassifier(max_depth=7), 
    "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4), 
    "xgboost"      : XGBClassifier(),
    "Neural Net"   : MLPClassifier(alpha=1), 
    "AdaBoost"     : AdaBoostClassifier(),
    "Naive Bayes"  : GaussianNB(), 
    "QDA"          : QuadraticDiscriminantAnalysis(),
    "Linear SVC"   : LinearSVC(),
    "Linear SVM"   : SVC(kernel="linear"), 
    "Gaussian Proc": GaussianProcessClassifier(1.0 * RBF(1.0)),
}
from time import time
nfast = 10      # Run the first nfast learner. Don't run the very slow ones at the end
head = list(classifiers.items())[:nfast]

for name, classifier in head:
    start = time()                     # remember starting training time
    classifier.fit(X_train, y_train)
    train_time = time() - start        # get the total training time
    start = time()
    score = classifier.score(X_test, y_test)
    score_time = time()-start         # get the score time
    print("{:<15}| score = {:.3f} | time = {:,.3f}s/{:,.3f}s".format(name, score, train_time, score_time))

#Dummy          | score = 0.193 | time = 0.000s/0.011s
#KNN(3)         | score = 0.547 | time = 6.505s/279.467s
#RBF SVM        | score = 0.583 | time = 1,401.787s/170.731s
#Decision Tree  | score = 0.608 | time = 5.983s/0.060s
#Random Forest  | score = 0.584 | time = 0.256s/0.057s
#XGBoost        | score = 0.636 | time = 423.858s/0.324s
#Neural Net     | score = 0.594 | time = 52.310s/0.064s
#AdaBoost       | score = 0.633 | time = 79.139s/2.731s
#Naive Bayes    | score = 0.174 | time = 1.429s/1.139s
#QDA            | score = 0.384 | time = 75.755s/3.419s
#Linear SVC     | score = 0.496 | time = 297.832s/0.028s

# Fitting AdaBoost Classification to the Training set
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=200, learning_rate=0.8)
start = time()  
classifier.fit(X_train, y_train)
train_time = time() - start 
start = time()  
score = classifier.score(X_test, y_test)
score_time = time() - start 
print("score = {:.3f} | time = {:,.3f}s/{:,.3f}s".format(score, train_time, score_time))

# Computing feature inportance
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_name = np.array(feature_name)
for f in range(100):
    print("%2d) %-*s %f" % (f + 1, 30, feature_name[indices[f]], importances[indices[f]]))
# 1) love                           0.024554
# 2) great                          0.020171
# 3) dryer                          0.017196
# 4) hair                           0.017130
# 5) return                         0.016330
# 6) month                          0.015915
# 7) easi                           0.012854
# 8) good                           0.012749
# 9) one                            0.011392
#10) use                            0.010875
#11) nice                           0.009870
#12) stop                           0.009578
#13) disappoint                     0.009571
#14) like                           0.009213
#15) work                           0.008348
#16) job                            0.008327
#17) time                           0.006925
#18) year                           0.006767
#19) product                        0.006757
#20) well                           0.006740
#21) cord                           0.006666
#22) excel                          0.006308
#23) hot                            0.006112
#24) wast                           0.006013
#25) perfect                        0.005845
#26) last                           0.005843
#27) dri                            0.005582
#28) far                            0.005379
#29) best                           0.005307
#30) littl                          0.005291
#31) complaint                      0.005248
#32) enough                         0.005204
#33) back                           0.005136
#34) spark                          0.005114
#35) would                          0.005083
#36) bit                            0.004915
#37) much                           0.004785
#38) howev                          0.004697
#39) look                           0.004494
#40) turn                           0.004450
#41) junk                           0.004428
#42) air                            0.004368
#43) realli                         0.004347
#44) fast                           0.004312
#45) fine                           0.004247
#46) even                           0.004077
#47) recommend                      0.004065
#48) get                            0.004027
#49) bought                         0.004002
#50) heat                           0.003988

# Visualization feature inportance
feature_name = cv.get_feature_names()
feature_name = np.array(feature_name)
tmp={"feature_name" : feature_name,
     "importances" : importances}
tmp_df=DataFrame(tmp)
tmp_df.sort_values("importances",ascending=False,inplace=True)

plt.figure(figsize=(20, 10))
plt.title("Words importance ranking (Top 50)", fontsize = 50)
plt.xticks(rotation=90)
sns.barplot(x=tmp_df["feature_name"][:50], y=tmp_df["importances"][:50], data=tmp_df)  
