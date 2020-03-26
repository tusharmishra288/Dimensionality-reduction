
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as sc
from sklearn.decomposition import PCA
from scipy import linalg as alg
r=pd.read_csv('bank_contacts.csv')
r_std=sc().fit_transform(r)
"""
method to find covariance matrix manually
mean_vec=np.mean(r_std,axis=0)
cov_mat=(r_std-mean_vec).T.dot((r_std-mean_vec)) / (r_std.shape[0]-1)
print("covariance matrix:",cov_mat)
"""
#by function
cov_mat=np.cov(r_std.T)
eig_val,eig_vec=alg.eig(cov_mat)
print(eig_val)
print(eig_vec)
#for visually confirming eigen pairs
eig_pairs=[(np.abs(eig_val[i]),eig_vec[:,i]) for i in range(len(eig_val))]
for i in eig_pairs:
    print(i[0])
 
t=PCA(n_components=2)
t.fit(r)
print(t.explained_variance_ratio_)
##for cumulative variance
#print(t.explained_variance_ratio_.sum())
u=t.fit(r_std)
f=t.transform(r_std)
print(f.shape)
print(r_std.shape)
plt.plot(np.cumsum(t.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.figure()
plt.scatter(f[:,0],f[:,1],cmap='plasma')
plt.xlabel('First pc')
plt.ylabel('Second pc')
plt.show()   