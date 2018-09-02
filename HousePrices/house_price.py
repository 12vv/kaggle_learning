import pandas as pd
import numpy as np 
import os

# visualization
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# % matplotlib inline


base_path = os.path.abspath(os.path.dirname(__file__))


train_df = pd.read_csv(base_path + '/data/train.csv')


# print(train_df['SalePrice'].describe())


# sns.distplot(train_df['SalePrice'])
# sns.distplot(train_df['YearBuilt'])

var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

plt.show()