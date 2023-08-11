#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import numpy as np

house = pd.read_csv('housing.csv')

house.head()


# ### Understanding the Data

# In[3]:


house.shape


# In[4]:


house.info()


# In[5]:


house.describe()


# In[6]:


train_set, test_set = train_test_split(house, test_size=0.2, random_state=2)
train_set.info()
train_set.shape


# In[8]:


data = train_set.copy()

data.plot(kind="scatter", x="longitude", y="latitude",
          s=data["population"]/30, label="population",
          c=data["median_house_value"], cmap=plt.get_cmap("jet"),
          figsize=(10,7), alpha=0.2,)


# In[9]:


import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


# In[22]:


data.info()


# In[23]:


corr_matrix = data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[24]:


features = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "households",
            "total_bedrooms", "population"]
scatter_matrix(data[features], figsize=(15,10))
plt.show()


# In[25]:


data.plot(kind="scatter", x="median_income", y="median_house_value", figsize=(10,7), alpha=0.4,)


# In[ ]:





# # Data Cleaning
# 1. ocean_proximity is string and we need to change to number and then need to drop the string. 
# 2. There is missing data in total bedroom. So, we need to fill the missing values. 

# In[10]:


ocean_proximity = 'ocean_proximity'

ocean_proximity_number = pd.get_dummies(data[ocean_proximity], prefix=ocean_proximity)

data = pd.concat([data, ocean_proximity_number], axis=1)

data.drop(columns=[ocean_proximity], inplace=True)


# In[12]:


data.head()


# In[13]:


median_total_bedrooms = data['total_bedrooms'].median()
data['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
data.info()


# 

# In[20]:


x = data.drop(columns=['median_house_value'])
x.head()
y= data['median_house_value']
y.head()


# In[23]:


from sklearn.linear_model import LinearRegression
OLS = LinearRegression()
OLS.fit(x,y)


# In[27]:


print(OLS.intercept_)
print(OLS.coef_)
print(OLS.score(x,y))


# In[54]:


y_pred = OLS.predict(x)
performance = pd.DataFrame({'PREDICTION': y_pred, 'ACTUAL VALUES': y})
performance['error'] = performance['ACTUAL VALUES']-performance['PREDICTION']
performance.head()


# In[55]:


performance.reset_index(drop=True, inplace=True)
performance.reset_index(inplace=True)
performance.head()

fig = plt.figure(figsize=(10,5))
plt.bar('index', 'error', data=performance, color= 'black', width=0.3)
plt.show()


# In[56]:


import statsmodels.api as sm
x = sm.add_constant(x)
nicerOLS = sm.OLS(y, x).fit()
nicerOLS.summary()


# In[ ]:




