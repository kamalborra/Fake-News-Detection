
#--------------------------------------------------------------
# Include Libraries
#--------------------------------------------------------------

import pandas as pd
import pandas

print(pandas.__version__)
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn import metrics
#from pandas_ml import ConfusionMatrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import numpy as np
import re
import csv
import pickle


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

#--------------------------------------------------------------
# Importing dataset using pandas dataframe
#--------------------------------------------------------------
df = pd.read_csv("fake_or_real_news.csv")
    
# Inspect shape of `df` 
df.shape

# Print first lines of `df` 
df.head()

# Set index 
df = df.set_index("Unnamed: 0")

# Print first lines of `df` 
df.head()
print(df.head())



#--------------------------------------------------------------
# Separate the labels and set up training and test datasets
#--------------------------------------------------------------
y = df.label 
df.drop("label", axis=1)      #where numbering of news article is done that column is dropped in dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)


with open('Train-SetX.csv','w',encoding='utf-8',newline='') as file:
	writer = csv.writer(file, delimiter=',')
	for line in X_train:
		print(line)
		writer.writerow([line])
		

with open('Test-SetX.csv','w',encoding='utf-8',newline='') as file:
	writer = csv.writer(file, delimiter=',')
	for line in X_test:
		writer.writerow([line])
		
with open('Train-SetY.csv','w',encoding='utf-8',newline='') as file:
	writer = csv.writer(file, delimiter=',')
	for line in y_train:
		writer.writerow([line])
		
		
with open('Test-SetY.csv','w',encoding='utf-8',newline='') as file:
	writer = csv.writer(file, delimiter=',')
	for line in y_test:
		writer.writerow([line])


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)                  # Learn the vocabulary dictionary and return term-document matrix.
count_test = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # This removes words which appear in more than 70% of the articles
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)


n_vect = CountVectorizer(min_df = 5, ngram_range = (1,2)).fit(X_train)
n_train = n_vect.fit_transform(X_train)
n_test = n_vect.transform(X_test)


#--------------------------------------------------------------
# Function to plot the confusion matrix 
#--------------------------------------------------------------

def plot_confusion_matrix(cm, classes,normalize=False,title='',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()





#--------------------------------------------------------------
# Naive Bayes classifier for Multinomial model 
#-------------------------------------------------------------- 
def NaiveBayes(xtrain,ytrain,xtest,ytest,ac):
	clf = MultinomialNB(alpha=.01, fit_prior=True)
	clf.fit(xtrain, ytrain)
	pred = clf.predict(xtest)
	score = metrics.accuracy_score(ytest, pred)
	print("accuracy:   %0.3f" % score)
	cm = metrics.confusion_matrix(ytest, pred, labels=['FAKE', 'REAL'])
	plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],title='Confusion matrix Naive Bayes')
	print(cm)
	ac.append(score)
	

def Logreg(xtrain,ytrain,xtest,ytest,ac):
    i=1
    logreg = LogisticRegression(C=9)
    logreg.fit(xtrain,ytrain)
    pred = logreg.predict(xtest)
    score = metrics.accuracy_score(ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(ytest, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],title='Confusion matrix Logistic')
    print(cm)
    ac.append(score)
	
def RForest(xtrain,ytrain,xtest,ytest,ac):
    clf1 = RandomForestClassifier(max_depth=50, random_state=0,n_estimators=25)
    clf1.fit(xtrain,ytrain)
    pred = clf1.predict(xtest)
    score = metrics.accuracy_score(ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(ytest, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],title='Confusion matrix RForest')
    print(cm)
    ac.append(score)
	

def SVM(xtrain,ytrain,xtest,ytest,ac):
    clf3 = SVC(C=100, gamma=0.1)
    clf3.fit(xtrain, ytrain)
    pred = clf3.predict(xtest)
    score = metrics.accuracy_score(ytest, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(ytest, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'],title='Confusion matrix SVM')
    print(cm)
    ac.append(score)
    


def process(xtrain,ytrain,xtest,ytest,ac):
	print("For Multinomial Naive BayesModel")
	NaiveBayes(xtrain,ytrain,xtest,ytest,ac)
    
	print("For Random Forest Classifiers")   
	RForest(xtrain,ytrain,xtest,ytest,ac)
    
	print("For Support Vector Machine_Radial Basis Function Classifier")
	SVM(xtrain,ytrain,xtest,ytest,ac)
    
	print("For Logarithamic Classifier")
	Logreg(xtrain,ytrain,xtest,ytest,ac)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
explode = (0.1, 0, 0, 0, 0)  

al=["NaiveBayes","Random Forest","LogisticRegressio","SVM"]
cac=[]
tac=[]
nac=[]

process(count_train,y_train,count_test,y_test,cac)
print(cac)

result2=open('CountAccuracy.csv', 'w')
result2.write("Algorithm,Accuracy" + "\n")
for i in range(0,len(cac)):
    print(al[i]+","+str(cac[i]))
    result2.write(al[i] + "," +str(cac[i]) + "\n")
result2.close()


fig = plt.figure(0)
df =  pd.read_csv('CountAccuracy.csv')
acc = df["Accuracy"]
alc = df["Algorithm"]
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Count Accuracy Value')
fig.savefig('CountAccuracy.png')
plt.show()

process(tfidf_train,y_train,tfidf_test,y_test,tac)
print(tac)

result2=open('TfidfAccuracy.csv', 'w')
result2.write("Algorithm,Accuracy" + "\n")
for i in range(0,len(cac)):
    print(al[i]+","+str(cac[i]))
    result2.write(al[i] + "," +str(tac[i]) + "\n")
result2.close()


fig = plt.figure(0)
df =  pd.read_csv('TfidfAccuracy.csv')
acc = df["Accuracy"]
alc = df["Algorithm"]
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Tfidf Accuracy Value')
fig.savefig('TfidfAccuracy.png')
plt.show()

process(n_train,y_train,n_test,y_test,nac)
print(nac)

result2=open('NgramAccuracy.csv', 'w')
result2.write("Algorithm,Accuracy" + "\n")
for i in range(0,len(cac)):
    print(al[i]+","+str(cac[i]))
    result2.write(al[i] + "," +str(nac[i]) + "\n")
result2.close()


fig = plt.figure(0)
df =  pd.read_csv('NgramAccuracy.csv')
acc = df["Accuracy"]
alc = df["Algorithm"]
plt.bar(alc, acc, align='center', alpha=0.5,color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Ngram Accuracy Value')
fig.savefig('NgramAccuracy.png')
plt.show()

#--------------------------------------------------------------
# Creating pickle files 
#--------------------------------------------------------------

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
pickle.dump(tfidf_vectorizer, open('C:\\Users\\kamal\\OneDrive\\Desktop\\Fake News Detection\\pickles\\pre_training.pkl', 'wb'))

logreg = LogisticRegression(C=9)
logreg.fit(tfidf_train,y_train)
pickle.dump(logreg, open('C:\\Users\\kamal\\OneDrive\\Desktop\\Fake News Detection\\pickles\\model.pkl', 'wb'))