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

# var = 'GrLivArea'
# var = 'YearBuilt'
# data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(train_df[cols], size=2.5)

# plt.show()

# missing data
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
