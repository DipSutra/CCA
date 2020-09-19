# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:27:13 2020

@author: Dipankar
"""
import pandas as pd
import numpy as np
data = pd.read_csv('StudentData.csv')
data = data[['Rigorous Instruction %',
      'Collaborative Teachers %',
     'Supportive Environment %',
       'Effective School Leadership %',
   'Strong Family-Community Ties %',
    'Trust %','Average ELA Proficiency',
       'Average Math Proficiency']]
# drop missing values
data = data.dropna()
# separate X and Y groups
X = data[['Collaborative Teachers %',
     'Supportive Environment %',
   'Strong Family-Community Ties %',
    'Trust %']]
      
Y = data[['Average ELA Proficiency',
       'Average Math Proficiency']]
for col in X.columns:
    X[col] = X[col].str.strip('%')
    X[col] = X[col].astype('int')
# Standardise the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=True, with_std=True)
X_sc = sc.fit_transform(X)
Y_sc = sc.fit_transform(Y)
import rcca
nComponents = 2 # min(p,q)=2
cca = rcca.CCA(kernelcca = False, reg = 0., numCC = nComponents,)
# train on data
cca.train([X_sc, Y_sc])
print('Canonical Correlation Per Component Pair:',cca.cancorrs)
print('% Shared Variance:',cca.cancorrs**2)