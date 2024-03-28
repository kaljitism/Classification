#!/usr/bin/env python
# coding: utf-8

# ## Cross Validation

# In[1]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[2]:


iris = load_iris()


# In[3]:


x, y = iris.data, iris.target


# In[4]:


knn = KNeighborsClassifier(n_neighbors=5)
score = cross_val_score(knn, x, y, cv=10, scoring='accuracy')


# In[5]:


score


# In[6]:


score.mean()


# ## HyperParameter Tuning

# In[14]:


from sklearn.model_selection import GridSearchCV


# In[15]:


param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11]}


# In[16]:


est = KNeighborsClassifier()


# In[17]:


ht = GridSearchCV(estimator=est, param_grid=param_grid, cv=10, scoring='accuracy')


# In[18]:


ht.fit(x, y)


# In[19]:


ht.best_params_


# In[20]:


ht.best_score_

