#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[3]:


from sklearn.preprocessing import scale
winedata=pd.read_csv('Downloads//wine.csv')


# In[4]:


##Standardize winedata using scale function
wine_norm=scale(winedata)
wine_norm.shape


# In[5]:


pca=PCA(n_components=14) ##No of features=14
pca_values=pca.fit_transform(wine_norm)


# In[6]:


var1=pca.explained_variance_ratio_
var1


# In[ ]:


##Percentage of varaiance in each and evry column (column 1 to 14).39% OF data is contained in 1st column


# In[7]:


pca.components_[1]


# In[ ]:


##PCA components of first feature(PCA1)


# In[8]:


var22 = np.cumsum(np.round(var1,decimals = 4)*100)
var22


# In[ ]:


##Cumulative variance of all the features show that 97 percent of data is contained in first 11 features. So, we need not consider
##last three features


# In[9]:


import seaborn as sns
plt.plot(var22,color="red")


# In[ ]:


##Line plot showing cumulative variance increasing with advancement of every feature


# In[10]:


##Constructing a biplot
x = pca_values[:,0]
y = pca_values[:,1]
plt.scatter(x,y)


# In[13]:


##The above scatterplot tells us that there is not high corelation and collinearity between PCA scores of feature 1 and 2.


# In[14]:


##Considering 1st 11 columns from the PCA TABLE:-
pca_new=pca_values[:,0:10]


# In[15]:


##Performing clustering using kmeans on pca scores
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
pca_brandnew=pd.DataFrame(pca_new)


# In[16]:


k=list(range(2,16))
TWSS= []


# In[17]:


for i in k:
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(pca_brandnew)
    WSS= []
    for j in range (i):
         WSS.append(sum(cdist(pca_brandnew.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,pca_brandnew.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# In[18]:


##Scree plot
plt.plot(k,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[19]:


###optimal k value=6
model=KMeans(n_clusters=6)
model.fit(pca_brandnew)
model.labels_
kclust_pca=pd.Series(model.labels_)


# In[20]:


##Performing k-means clustering on wine data

wine_new=pd.DataFrame(wine_norm)


# In[21]:


kk=list(range(2,16))
TWSS1= []
for n in kk:
    kmeans1= KMeans(n_clusters=n)
    kmeans1.fit(wine_new)
    WSS1= []
    for m in range (n):
         WSS1.append(sum(cdist(wine_new.iloc[kmeans1.labels_==m,:],kmeans1.cluster_centers_[m].reshape(1,wine_new.shape[1]),"euclidean")))
    TWSS1.append(sum(WSS1))


# In[22]:


plt.plot(kk,TWSS,'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# In[23]:


###optimal k value=6
model22=KMeans(n_clusters=6)
model22.fit(wine_new)
kclust_wine=pd.Series(model22.labels_)


# In[ ]:


##Thus, after performing k-means clustering on pca scores and wine data we same
##same optimal number of clusters.


# In[25]:


##Performing hierarchal clustering on pca scores
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
pca_hier=linkage(pca_brandnew,method='complete',metric='euclidean')
pca1_hier=AgglomerativeClustering(n_clusters=6,linkage='complete',affinity='euclidean').fit(pca_brandnew)
pca1_hier.labels_
hclust_pca=pd.Series(pca1_hier.labels_)


# In[29]:


##Performing hierarchal clustering on wine_data
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
pca_hier1=linkage(wine_new,method='complete',metric='euclidean')
pca2_hier=AgglomerativeClustering(n_clusters=6,linkage='complete',affinity='euclidean').fit(wine_new)
pca2_hier.labels_
hclust_wine=pd.Series(pca2_hier.labels_)


# In[32]:





# In[36]:


##No. of unique clusters for kmeans and hierarchichal for pca wine data and normal data
analysis.columns=['kclust_pca','kclust_wine','hclust_pca','hclust_wine']
a=analysis.kclust_pca.value_counts()
b=analysis.kclust_wine.value_counts()
c=analysis.hclust_pca.value_counts()
d=analysis.hclust_wine.value_counts()


# In[38]:


analysis=pd.concat([a,b,c,d],axis=1)


# In[39]:


analysis


# In[ ]:





# In[ ]:





# In[ ]:




