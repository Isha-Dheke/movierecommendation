#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies = movies.merge(credits, on ='title')


# In[6]:


movies.shape


# In[7]:


movies.head()


# In[8]:


movies = movies[['id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.isnull().sum()


# In[10]:


movies.info()


# In[11]:


movies.dropna(inplace = True)


# In[12]:


movies.isnull().sum()


# In[13]:


movies.duplicated().sum()


# In[14]:


movies.iloc[0].genres


# In[15]:


import ast


# In[16]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[17]:


movies['genres'] = movies['genres'].apply(convert)


# In[18]:


movies.head()


# In[19]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[20]:


movies.head()


# In[21]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
    return L


# In[22]:


movies['cast'] = movies['cast'].apply(convert3)


# In[23]:


movies.head()


# In[24]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[25]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[26]:


movies.head()


# In[27]:


movies.shape


# In[28]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[29]:


movies.head()


# In[30]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[31]:


movies.head()


# In[32]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[33]:


movies.head()


# In[34]:


new_df = movies[['id','title','tags']]


# In[35]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[36]:


new_df.head()


# In[37]:


import nltk


# In[38]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[39]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[40]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[41]:


new_df


# In[42]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[43]:


new_df.head()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')


# In[45]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[46]:


vectors[0]


# In[47]:


cv.get_feature_names_out()


# In[48]:


from sklearn.metrics.pairwise import cosine_similarity


# In[49]:


similarity = cosine_similarity(vectors)


# In[50]:


similarity


# In[51]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[52]:


recommend('Batman Begins')


# In[54]:


recommend('Avatar')

