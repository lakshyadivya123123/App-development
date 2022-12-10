#!/usr/bin/env python
# coding: utf-8

# In[1]:


#panda task
import pandas as pd


# In[15]:



df=pd.read_csv('survey_results_public.csv')


# In[3]:


df.info()


# In[4]:



df.head()


# In[5]:


df.columns


# In[6]:


df = df['Employment']


# In[7]:



df.head(2)


# In[8]:


schema=pd.read_csv('survey_results_schema.csv')


# In[9]:



schema['question'][14]


# In[10]:



schema['type'][14]


# In[11]:


df.value_counts()


# In[12]:


divya=df.str.contains('Employed, full-time')


# In[13]:



divya.value_counts(normalize=True)


# In[16]:



df[df['Country'].str.contains('United states of America|United Kingdom of Great Britain and Northern Ireland', na = False)]


# In[ ]:





# In[ ]:





# In[ ]:




