#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df = pd.read_csv(path)


# In[4]:


df.head()


# ## Question 1

# In[5]:


print(df.dtypes)


# ## Question 2

# In[6]:


df.drop(['Unnamed: 0', 'id'], axis=1, inplace = True)
df.describe()


# ## Question 3

# In[7]:


df["floors"].value_counts().to_frame()


# ## Question 4

# In[8]:


sns.boxplot(x = "waterfront", y = "price", data = df)


# ## Question 5

# In[9]:


sns.regplot(x = "sqft_above", y = "price", data = df)


# ## Question 6

# In[10]:


X = df[['price']]
Y = df['sqft_living']

lm = LinearRegression()

lm.fit(X,Y)

print('The R-square is: ', lm.score(X, Y))


# ## Question 7

# In[ ]:


features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]     
X = df[features]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)


# ## Question 8

# In[ ]:


Input=[('scale', StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(X,Y)
pipe.score(X,Y)


# ## Question 9

# In[11]:


features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]     
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])


# In[ ]:


RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)


# ## Question 10

# In[ ]:


pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = Ridge(alpha=0.1)
poly.fit(x_train_pr, y_train)
poly.score(x_test_pr, y_test)

