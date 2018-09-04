import pandas as pd
import numpy as np 
import os

import seaborn as sns
import matplotlib.pyplot as plt


base_path = os.path.abspath(os.path.dirname(__file__))

train_df = pd.read_csv(base_path+'/data/train.csv')
test_df = pd.read_csv(base_path+'/data/test.csv')



combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# 缺失数据的处理
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            # 去掉所有 sex=i 且 pclass=j+1 且age缺失的行
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            # 统计这些 sex=i 且 pclass=j+1 的age的中位数
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

# correlation matrix
corrmat = train_df.corr()
# sns.heatmap(corrmat, vmax=.8, square=True)

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Survived')['Survived'].index

cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()