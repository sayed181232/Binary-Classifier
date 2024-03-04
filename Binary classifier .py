#!/usr/bin/env python
# coding: utf-8

# we will be using the MNIST dataset, which is a set of 70,000 small
# images of digits handwritten by high school students and employees of the US Cen‐
# sus Bureau. Each image is labeled with the digit it represents
# **using Binary classifier to know this image is 5 or not** 
# 

# In[2]:


# frist download data from mnist 
from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784",version=1,as_frame=False)
mnist.keys()


# In[3]:


#Let’s look at these arrays:
x,y=mnist["data"],mnist["target"]


# In[4]:


x.shape


# In[5]:


y.shape


# 
# 

# In[64]:


#plot some digit from mnist data 
import matplotlib.pyplot as plt
import matplotlib as mpl
some_digit=x[0]
some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show


# In[7]:


y[0]


# In[8]:


#ohhh the machine learning not handle str 
import numpy as np 
y=y.astype(np.uint8)


# In[9]:


y[0]


# In[10]:


#split the data into train and test 
x_train,y_train,x_test,y_test=x[:60000],y[:60000],x[60000:],y[60000:]


# In[11]:


y_train_5=(y_train==5)
y_test_5=(y_test==5)


# In[12]:


# let's go train The Binary Classifier 
#use SGDclassifier 
from sklearn.linear_model import SGDClassifier
SGD=SGDClassifier(random_state=42)
SGD.fit(x_train,y_train_5)


# In[13]:


#predict some_digit
SGD.predict([some_digit])


# In[14]:


# now we need to Measure Accuracy Using Cross-Validation 
from sklearn.model_selection import cross_val_score
cross_val_score(SGD,x_train,y_train_5,cv=3,scoring="accuracy")


# **WOW!** the accuracy is above 95% This looks amazing, doesn’t it? Well, before you get too excited, let’s look at a very
# dumb classifier that just classifies every single image in the “not-5” class:

# In[15]:


from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# **Can you guess this model’s accuracy?** Let’s find out:
# 

# In[16]:


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, x_train, y_train_5, cv=3, scoring="accuracy")


# **That’s right, it has over 90% accuracy!** This is simply because only about 10% of the
# images are 5s, so if you always guess that an image is not a 5, you will be right about
# 90% of the time. Beats Nostradamus.
# 

# A much better way to evaluate the performance of a classifier is to look at **the confu‐
# sion matrix**

# In[17]:


from sklearn.model_selection import cross_val_predict
y_predict=cross_val_predict(SGD,x_train,y_train_5,cv=3)


# In[18]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5,y_predict)


# In[19]:


perfect_y_train=y_train_5
confusion_matrix(y_train_5,perfect_y_train)


# In[20]:


#now we want to calculate the recall and precision
from sklearn.metrics import precision_score , recall_score 
precision_score(y_train_5,y_predict)


# In[21]:


recall_score(y_train_5,y_predict)


# **Precision/Recall Tradeoff**
# 
# 

# In[22]:


#We want to predict by threshuld 
y_score=SGD.decision_function([some_digit])
print(y_score)


# In[23]:


threshuld=0
#if y_score > threshuld this image is 5 
y_predict_th=y_score>threshuld
print(y_predict_th)


# In[24]:


# we change in threshuld 
threshuld=8000
y_predict_th=y_score>threshuld
print(y_predict_th)


# **you must choise the best threshuld**. But How?
# 

# solution is plot realtion  between recall and precision and select the intersection 

# In[25]:


y_scores = cross_val_predict(SGD, x_train, y_train_5, cv=3,
 method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# In[26]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    fig,ax=plt.subplots()
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.set_xlabel("thresholds")
    leg=ax.legend()
    [...] # highlight the threshold, add the legend, axis label and grid
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()


# **We are on the steps to finish the project**

# **now we used the ROC curve to select the best model train**

# In[27]:


from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    fig,ax=plt.subplots()
    ax.set_xlabel("False positive rater")
    ax.set_ylabel("True positive rater")
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    #leg=ax.legend()
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
plt.show()


# **One way to compare classifiers is to measure the area under the curve (AUC). A per‐
# fect classifier will have a ROC AUC equal to 1**

# In[28]:


from sklearn.metrics import roc_auc_score
AUC=roc_auc_score(y_train_5,y_scores)
print("Area under the curve equal: ",AUC)


# **Let’s train a RandomForestClassifier and compare its ROC curve and ROC AUC
# score to the SGDClassifier**

# In[58]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3,
 method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)


# In[59]:


fig,ax=plt.subplots()
ax.set_xlabel("False positive rater")
ax.set_ylabel("True positive rater(Recall)")

plt.plot(fpr, tpr, "b:", label="SGD")
plt.plot(fpr_forest, tpr_forest,"r-", label="Random Forest")
plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
leg=ax.legend()
#plt.legend(loc="lower right")
plt.show()


# In[57]:


AUC_forest=roc_auc_score(y_train_5,y_scores_forest)
print("Area under the curve equal: ",AUC_forest)


# **WOW it's better then SGD**

# let's go calculate the precision and recall

# In[61]:


y_probas_forest = cross_val_predict(forest_clf, x_train, y_train_5, cv=3)
precision=precision_score(y_train_5,y_probas_forest)
print("precision of random forest equal:",precision)
recall=recall_score(y_train_5,y_probas_forest)
print("recall of random forest equal:",recall)


# **Try measuring the precision and recall scores: you should find 99.0% precision and
# 86.6% recall. Not too bad!**
# 

# In[ ]:




