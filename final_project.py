import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
rcParams["figure.figsize"] =10,5
%matplotlib inline

data = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv",dtype={"title": str, "removed_by": object})[['title','removed_by']]
vocab = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv")[['title']]


stopwords = []
with open ("../input/english-stopwords/stop-word-list.csv") as f:
    for j in f:
        for i in j.split(','):
            stopwords.append(i)


test_size = 0.3
random_state = 0
target = pd.read_csv("../input/dataisbeautiful/r_dataisbeautiful_posts.csv")[['removed_by']]

target.replace(to_replace='moderator' ,
        value=True,
        inplace=True,
        limit=None,
        regex=False, 
        method='pad'
              )

target.replace(to_replace='deleted' ,
        value=True,
        inplace=True,
        limit=None,
        regex=False, 
        method='pad'
              )
target.replace(to_replace='automod_filtered' ,
        value=True,
        inplace=True,
        limit=None,
        regex=False, 
        method='pad'
              )
target.replace(to_replace='reddit' ,
        value=True,
        inplace=True,
        limit=None,
        regex=False, 
        method='pad'
              )
target.fillna(value = False,
              axis = 1, 
              inplace = True, 
              limit = None, 
              downcast = None
             )


vocabulary = []
decode_str = ""

voc ={}
n = 0
for i in data.values.tolist():
    for k in i:
        if type(k) == str:
            for j in k.split():
                if j not in voc:
                    if j not in stopwords:
                        decode_str = j.encode("ascii","ignore")
                        #vocabulary.append(j)
                        voc[decode_str.decode()] = n
                        n+=1


m = len(voc)
augmented_input =[]
temp = [0]*m
n=0
for i in data.values.tolist():
    for k in i:
        if type(k) ==str:
            for j in k.split():
                if j not in stopwords:
                    decode_str = j.encode("ascii","ignore")
                    temp.append(voc[decode_str.decode()])
    augmented_input.append(temp)
    
foo = []
for i in target[:10].values.ravel():
    foo.append(i)  


#splitting data - DO NOT RUN WITH FULL SET UNLESS YOU HAVE SEVERAL DAYS TO SPARE!
#x_train, x_test, y_train, y_test = train_test_split(augmented_input, target, test_size = test_size, random_state = random_state)

#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[0:10], foo, test_size = test_size, random_state = random_state)


n_iter =10  #iterations
eta0 =0.1 #learning rate


perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)

#predictions (change to desired test)
y_pred = perc.predict(x_test)


print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))

#n = 100
#change target values into something sclearn likes 
foo = []
for i in target[:100].values.ravel():
    foo.append(i)
    
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[0:100], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
#predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))

#n = 200
foo = []
for i in target[:200].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[:200], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
# predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n=300
foo = []
for i in target[200:500].values.ravel():
    foo.append(i)
    
    
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[200:500], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
# predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n=400
foo = []
for i in target[500:900].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[500:900], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
# predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n = 500
foo = []
for i in target[900:1400].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[900:1400], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)

y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n = 1000
foo = []
for i in target[1400:2400].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[1400:2400], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
# predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n = 1500
#change target values into something sclearn likes 
foo = []
for i in target[2500:4000].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[2500:4000], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
# predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))


#n = 5000
#change target values into something sclearn likes 
foo = []
for i in target[:5000].values.ravel():
    foo.append(i)
#splitting much smaller data
x_train, x_test, y_train, y_test = train_test_split(augmented_input[:5000], foo, test_size = test_size, random_state = random_state)
perc = Perceptron(n_iter_no_change=n_iter,eta0=eta0,random_state=random_state)
perc.fit(x_train,y_train)
#predictions
y_pred = perc.predict(x_test)
print("accuracy: {0:.2f}%".
         format(accuracy_score(y_test,
                               y_pred)*100))
