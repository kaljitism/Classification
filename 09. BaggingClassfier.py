#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dependencies
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# In[2]:


# Data Extraction
cancer_data = datasets.load_breast_cancer()                       


# In[3]:


# Keys
cancer_data.keys()


# In[4]:


# Data Attributes
cancer_data.feature_names


# In[5]:


# Output Class
cancer_data.target_names


# In[6]:


# Description
print(cancer_data.DESCR)


# In[7]:


# Getting features and labels in x and y
x, y = cancer_data.data, cancer_data.target


# In[8]:


# Splitting up into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[9]:


bagging = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=100)


# In[10]:


# Fitting the data
bagging.fit(x_train, y_train)


# In[11]:


# Predictions
predictions = bagging.predict(x_test)


# In[12]:


predictions


# In[13]:


# Confusion Matrix
confusion_matrix(y_test, predictions)


# In[14]:


# Accuracy
bagging.score(x_test, y_test)


# In[15]:


# Precision
precision_score(y_test, predictions)


# In[16]:


# Recall
recall_score(y_test, predictions)


# In[17]:


# F1 Score
f1_score(y_test, predictions)


# In[18]:


y_pred_prob = bagging.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


# In[19]:


plt.plot([0, 1], [1, 0], 'k--')
plt.plot(fpr, tpr, label='Bagging')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Bagging ROC Curve')
plt.show()

