#!/usr/bin/env python
# coding: utf-8

# # Importing the modules

# In[38]:


import pandas as pd #Reading and manipulating data series

#For performing mathematical and statistical operations
import numpy as np
import statistics

#For data visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import plotly.express as px

#For data preoaration for model training using machine learning algorithms
from sklearn import preprocessing

#For estimating model performences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Suppressing warnings
import warnings
warnings.filterwarnings('ignore')


# # Loading the data

# In[16]:


dat=pd.read_csv('API_19_DS2_en_csv_v2_5346672_final.csv')
dat


# # Data Preprocessing I

# ## Checking for NaN values

# In[17]:


dat.isnull().sum()


# In[18]:


dat = dat.fillna(dat.mean())


# In[19]:


dat.isnull().sum()


# # Data Analysis I

# ## Data Visualization

# In[47]:


dat.plot.scatter(x = '2017', y = '1960');


# In[37]:


sns.lineplot(data=dat, x="1960", y="1961")


# In[46]:


corr = dat.corr().abs()

f, ax = plt.subplots(figsize=(12, 10))

mask = np.triu(np.ones_like(corr, dtype=bool))

hmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, annot=True, mask = mask, cmap=hmap)


# In[50]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
   
np.random.seed(10**7)
mu = 121 
sigma = 21
x = mu + sigma * np.random.randn(1000)
   
num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = 1, 
                            color ='green',
                            alpha = 0.7)
   
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
  
plt.plot(bins, y, '--', color ='black')
  
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
  
plt.title('matplotlib.pyplot.hist() function Example\n\n',
          fontweight ="bold")
  
plt.show()


# # Data Preprocessing II

# ## Checking for inoperative data types 

# In[23]:


dat.dtypes


# In[24]:


dat[['Country Name']]=dat[['Country Name']].apply(lambda col:pd.Categorical(col).codes)


# In[25]:


dat[['Country Code']]=dat[['Country Code']].apply(lambda col:pd.Categorical(col).codes)


# In[26]:


dat[['Indicator Name']]=dat[['Indicator Name']].apply(lambda col:pd.Categorical(col).codes)


# In[27]:


dat[['Indicator Code']]=dat[['Indicator Code']].apply(lambda col:pd.Categorical(col).codes)


# In[28]:


dat.dtypes


# # Data Analysis II

# ## Statistical Analysis

# In[29]:


dat.describe()


# In[30]:


dat.corr().abs()


# In[35]:


# The "2018" and "2017" variables share highest correlation-coefficient,
# thus the nature of the regression line formed from a regression equation involving these two variable is linear
# Here is the implementation of the above claim
plt.plot(dat['2017'], dat['2018'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




