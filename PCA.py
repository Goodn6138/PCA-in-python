import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

#generte data

genes = ['gene'+str(i) for i in range (1,101)]
wt = ['wt'+str(i) for i in range (1,6)]
ko = ['ko'+str(i) for i in range (1,6)]

data = pd.DataFrame(columns = [*wt , *ko], index = genes)
for gene in data.index:
    data.loc[gene , 'wt1':'wt5'] = np.ones(5)#np.random.poisson(lam = rd.randrange(10 , 1000) ,size= 5)
    data.loc[gene , 'ko1':'ko5'] = np.random.randn(5)#np.random.poisson(lam = rd.randrange(10 , 1000) ,size= 5)

print(data.head())
#_____PCA
#samples are columns
scaled_data = preprocessing.scale(data.T)
pca = PCA()

pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_*100 , decimals = 1)
labels = ["PC"+str(x) for x in range(1 , len(per_var)+1)]
print(per_var)
plt.bar(x = range(1 , len(per_var)+1) , height = per_var , tick_label = labels)
plt.title('Scree Plot')
plt.xlabel('Principle Component')
plt.ylabel('Percentage of Explained Variance')
plt.show()

loading_scores = pd.Series(pca.components_[0], index = genes)
sorted_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_scores[0 : 10].index.values
print(loading_scores[top_10_genes])
