from flask import Flask,render_template,request,url_for

#EDA Packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
df=pd.read_csv(r"https://raw.githubusercontent.com/RohithKumar1999/stockcsv/master/data.csv",encoding="ISO-8859-1")
data=df.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
for index in new_Index:
    data[index]=data[index].str.lower()
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
countvector=CountVectorizer(ngram_range=(1,1))
dataset=countvector.fit_transform(headlines)
X = dataset
y = df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=None,
            max_features=100, max_leaf_nodes=10,
            min_impurity_decrease=0.0, random_state=0,
            splitter='best')
clf.fit(X_train, y_train)
y_pred =  clf.predict(X_test)

accuracy_score(y_test, y_pred)
pickle.dump(DecisionTreeClassifier, open('model.pkl','wb'))
