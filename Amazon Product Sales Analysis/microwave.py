# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:21:50 2020

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
df_microwave = pd.read_csv("microwave.tsv", sep="\t", quoting = 3)

'''
---------------------------------1----------------------------------
'''

# Adding a length column for analyzing the length of the reviews
df_microwave['length'] = df_microwave['review_body'].apply(len)
#df_microwave.groupby('length').describe().sample(10)
#df_microwave.groupby('star_rating').describe()

## Adding a feedback column for analyzing the success based on star_rating
#df_microwave['feedback'] = 0
#df_microwave.loc[df_microwave.star_rating>3, 'feedback'] = 2
#df_microwave.loc[df_microwave.star_rating==3, 'feedback'] = 1
#df_microwave.loc[df_microwave.star_rating<3, 'feedback'] = 0
#df_microwave.groupby('feedback').describe()

# Visualization for distribution of star_rating for microwave
star_rating = df_microwave['star_rating'].value_counts()
label_rating = star_rating.index
size_rating = star_rating.values
colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']
rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'microwave', hole = 0.3)
df = [rating_piechart]
layout = go.Layout(title = 'Distribution of star_rating for microwave')
fig = go.Figure(data = df, layout = layout)
py.plot(fig)

# 3D-Visualization for realation of (1)length (2)star rating (3)product id
trace = go.Scatter3d(
    x = df_microwave['length'],
    y = df_microwave['star_rating'],
    z = df_microwave['product_id'],
    name = 'microwave',
    mode='markers',
    marker=dict(
        size=5,
        color = df_microwave['star_rating'],
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
df_microwave['length'].value_counts().plot.hist(color = color, figsize = (20, 10), bins = 20)
plt.title('Distribution of Review Length in Reviews', fontsize = 50)
plt.xlabel('review lengths', fontsize = 20)
plt.ylabel('count', fontsize = 20)
plt.xticks(np.arange(0, 15, step=1))
plt.show()

# Visualization for sales of different product
color = plt.cm.copper(np.linspace(0, 1, 50))
df_microwave['product_id'].value_counts().nlargest(50).plot.bar(color = color, figsize = (20, 10))
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
sns.boxplot(df_microwave['star_rating'], df_microwave['length'], palette = 'Blues')
plt.title("Relations between Review Length and Star Rating", fontsize = 50)
plt.show()

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')
plt.xlabel('star_rating', fontsize = 50)
plt.ylabel('review_length', fontsize = 50)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
sns.stripplot(df_hair_dryer['star_rating'], df_hair_dryer['length'], palette = 'Reds')
plt.title("Relations between Review Length and Star Rating", fontsize = 60)
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
for i in range(0, df_microwave.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df_microwave['review_body'][i])
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
df_microwave["review_date"] = pd.to_datetime(df_microwave["review_date"], format='%m/%d/%Y')
df_microwave['year'], df_microwave['month']= df_microwave['review_date'].dt.year, df_microwave['review_date'].dt.month

# Calculating avg_star_rating and total_comments
avg_star_rating = []
total_comments  = []
for i in range(min(df_microwave['year']), max(df_microwave['year'])+1):
    avg_star_rating.append((df_microwave[df_microwave['year'] == i]["star_rating"].sum())/(df_microwave[df_microwave['year'] == i].shape[0]))
    total_comments.append((df_microwave[df_microwave['year'] == i].shape[0]))

# Find avg_star_rating and total_comments for each year respectively
tmp={"avg_star_rating" : avg_star_rating,
     "total_comments" : total_comments}
tmp_df=DataFrame(tmp)
tmp_df["year"]=[2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]

# Visualize average star rating and comments number for each year
sns.set_style("whitegrid")
 
x  = tmp_df["year"]
y1 = tmp_df['total_comments']
y2 = tmp_df['avg_star_rating']

plt.rcParams['figure.figsize'] = (20,10)
fig = plt.figure()

plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

ax1 = fig.add_subplot(111)
ax1.bar(x, y1, alpha=.7, color='g')
ax1.set_ylabel('number_of_comments',fontsize='50')
ax1.set_title("Average Star Rating and Comments Number in 2004-2015",fontsize='60')

ax2 = ax1.twinx()
ax2.plot(x, y2, 'orange', ms=10, marker='*', alpha=0.5, linewidth=5)
ax2.set_ylabel('average_star_rating',fontsize='50')

'''
------------------------------Part c--------------------------------
'''

# Generate dataset with sales volumn, average star rating and reviews based on product ID
tmp_df1 = df_microwave.loc[:,'product_id'].value_counts()
grouped=df_microwave['star_rating'].groupby(df_microwave['product_id'])
tmp_df2 = grouped.mean()

tmp_df = pd.merge(tmp_df1, tmp_df2, how='left', left_index=True, right_index=True)
tmp_df.rename(columns={"product_id": "sales", "star_rating": "avg_star_rating"}, inplace=True)

tmp_df3 = DataFrame(cv.fit_transform(corpus).toarray())
tmp_df3["product_id"] = df_microwave["product_id"]
tmp_df3 = tmp_df3.groupby(by=["product_id"]).sum()
tmp_df = pd.merge(tmp_df, tmp_df3, how='left', left_index=True, right_index=True)

# Calibrating label for "success"
# tmp_df["sales"].describe()
tmp_df["success"] = 0
tmp_df.loc[tmp_df.sales>=13,'success'] = 1
tmp_df.loc[tmp_df.sales <13,'success'] = 0

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

#Dummy          | score = 0.562 | time = 0.000s/0.001s
#KNN(3)         | score = 0.750 | time = 0.002s/0.007s
#RBF SVM        | score = 0.312 | time = 0.023s/0.006s
#Decision Tree  | score = 0.938 | time = 0.008s/0.000s
#Random Forest  | score = 0.875 | time = 0.006s/0.001s
#xgboost        | score = 0.938 | time = 0.723s/0.002s
#Neural Net     | score = 0.875 | time = 3.970s/0.001s
#AdaBoost       | score = 0.946 | time = 0.278s/0.009s
#Naive Bayes    | score = 0.438 | time = 0.007s/0.001s
#QDA            | score = 0.562 | time = 0.024s/0.002s
    
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
# score = 0.938 | time = 1.421s/0.029s

# Calculating feature inportance
feature_name = cv.get_feature_names()
feature_name = np.array(feature_name)
feature_name = np.insert(feature_name, 0, "avg_star_rating", axis=0)
importances = classifier.feature_importances_
indices = np.argsort(importances)[::-1]
feature_name = np.array(feature_name)
for f in range(100):
    print("%2d) %-*s %f" % (f + 1, 30, feature_name[indices[f]], importances[indices[f]]))
# 1) work                           0.157300
# 2) great                          0.087510
# 3) one                            0.082989
# 4) like                           0.081697
# 5) get                            0.068404
# 6) got                            0.046669
# 7) well                           0.046561
# 8) use                            0.042970
# 9) product                        0.040532
#10) microwav                       0.024038
#11) door                           0.022800
#12) thought                        0.021277
#13) littl                          0.020120
#14) new                            0.014662
#15) howev                          0.014417
#16) place                          0.013357
#17) also                           0.013326
#18) bought                         0.012931
#19) easi                           0.012474
#20) good                           0.011852
    
# 1) disappoint                     0.000000
# 2) problem                        0.000000
# 4) probelm                        0.000000
# 5) nois                           0.000000
# 6) hazard                         0.000000
# 7) profil                         0.000000
# 8) worri                          0.000000
# 9) repair                         0.000000
#10) prompt                         0.000000
#11) quit                           0.000000
#12) sad                            0.000000
#13) rare                           0.000000
#14) pricey                         0.000000
#15) rash                           0.000000
#16) question                       0.000000
#17) prohibit                       0.000000
#18) program                        0.000000
#19) poorli                         0.000000
#20) short                          0.000000

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
pos_words="disappoint problem probelm nois hazard profil prompt repair worri quit sad rare pricey pricey rash question prolong prohibit program program profession poorli short"
wc = wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000)
wc.generate(pos_words)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Important Words for Failure (top 20)", fontsize = 50)
plt.show()

pos_words="work great sure like get easier well use smooth profession care perfect often new definit place also bought easi good"
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

data_df = df_microwave[["year", "star_rating"]]
                         
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (30, 18)
plt.title("Number of Reviews with Different Star Rating in 2004-2015", fontsize = 50)
sns.countplot(y="year", hue="star_rating", data=data_df, palette="Set2")
plt.show()

'''
------------------------------Part e--------------------------------
'''

# Build a model to find the most important words for star rating
# Load the data
X = cv.fit_transform(corpus).toarray()
y = df_microwave.loc[:, "star_rating"].values
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

#Dummy          | score = 0.189 | time = 0.000s/0.001s
#KNN(3)         | score = 0.446 | time = 0.247s/4.343s
#RBF SVM        | score = 0.424 | time = 22.489s/3.000s
#Decision Tree  | score = 0.492 | time = 0.344s/0.007s
#Random Forest  | score = 0.443 | time = 0.041s/0.009s
#xgboost        | score = 0.576 | time = 41.644s/0.026s
#Neural Net     | score = 0.539 | time = 24.179s/0.005s
#AdaBoost       | score = 0.514 | time = 4.749s/0.208s
#Naive Bayes    | score = 0.331 | time = 0.115s/0.084s
#QDA            | score = 0.139 | time = 0.456s/0.098s

# Fitting AdaBoost Classification to the Training set
# Adjusting parameters
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

#Fit the algorithm on the data
alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
    
#Predict training set:
dtrain_predictions = alg.predict(dtrain[predictors])
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
#Print model report:
print "\nModel Report"
print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                
feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

classifier = XGBClassifier(
                            learning_rate=0.01,
                            n_estimators=1000,
                            max_depth=8,
                            min_child_weight=2,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective= 'binary:logistic',
                            scale_pos_weight=1,
                            seed=27)
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

# 1) warrante                       0.009999
# 2) button                         0.007892
# 3) prize                          0.005033
# 4) serv                           0.004630
# 5) repaint                        0.004510
# 6) imporant                       0.004033
# 7) upon                           0.003976
# 8) center                         0.003947
# 9) honor                          0.003785
#10) awar                           0.003387
#11) kenmor                         0.003326
#12) rational                       0.003323
#13) percol                         0.003095
#14) sorri                          0.003084
#15) greasi                         0.003024
#16) peopl                          0.003020
#17) louver                         0.003005
#18) danbi                          0.002951
#19) domin                          0.002941
#20) preserv                        0.002938
#21) built                          0.002935
#22) pure                           0.002934
#23) whine                          0.002929
#24) copi                           0.002925
#25) flight                         0.002906
#26) struggl                        0.002893
#27) expens                         0.002887
#28) cheapi                         0.002870
#29) sometim                        0.002855
#30) easer                          0.002829
#31) quot                           0.002804
#32) regul                          0.002783
#33) sen                            0.002775
#34) custom                         0.002775
#35) exceedingli                    0.002766
#36) loud                           0.002763
#37) trigger                        0.002734
#38) yep                            0.002713
#39) esp                            0.002709
#40) firday                         0.002702
#41) ensur                          0.002678
#42) notebook                       0.002677
#43) retrofit                       0.002659
#44) standpoint                     0.002637
#45) unwil                          0.002626
#46) space                          0.002622
#47) starter                        0.002608
#48) har                            0.002606
#49) simpler                        0.002602
#50) klutz                          0.002585
    
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
