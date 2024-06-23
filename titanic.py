#!/usr/bin/env python
# coding: utf-8

# In[75]:


get_ipython().system('python3 titanic.py')


# In[ ]:


get_ipython().system('pip install -q japanize_matplotlib')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import japanize_matplotlib

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')

plt.rcParams['figure.figsize'] = 10, 10


# In[ ]:


df_train = pd.read_csv('/content/train.csv')
df_test = pd.read_csv('/content/test.csv')

train_mid = df_train.copy()
test_mid = df_test.copy()

# 後に分割するためフラグを振る
train_mid['train_or_test'] = 'train' #学習データフラグを追加
test_mid['train_or_test'] = 'test' #テストデータフラグを追加
test_mid['Survived'] = 9 #テストにSurvivedカラムを仮置き

df_all = pd.concat(
    [
        train_mid,
        test_mid
    ],
    sort=False,
    axis=0
).reset_index(drop=True)


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train.describe(include='all')


# In[ ]:


df_test.isnull().sum()


# In[ ]:


# 欠損値の確認
df_all.isnull().sum()


# In[ ]:


# Embarkedには最頻値を代入
df_all = df_all.fillna({'Embarked':df_all['Embarked'].mode()[0]})
# Fareには中央値を代入
df_all = df_all.fillna({'Fare':df_all['Fare'].median()})
# 欠損値補完がされたかを確認
df_all.isnull().sum()


# In[ ]:


# NameにMissが含まれているAgeの欠損値に15を追加
# df_all['honorific'] = df_all['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
# condition = (df_all['honorific'] == 'Miss') & (df_all['Age'].isna())
# df_all.loc[condition, 'Age'] = 15
# print(df_all.isnull().sum())
# 恣意的であるため却下


# In[ ]:


df_all.describe(include='all')


# In[ ]:


df_all['family_name'] = df_all['Name'].map(lambda y: y.split(', ')[0])
df_all.head()


# In[ ]:


# family_nameは階級を表すことができる？
df_all['family_name'].value_counts()


# In[ ]:


# Familysizeを定義
df_all['FamilySize'] = df_all['Parch'] + df_all['SibSp'] + 1
# FamilySizeを4種類に離散化
df_all['FamilySize_bin'] = 'big'
df_all.loc[df_all['FamilySize']==1,'FamilySize_bin'] = 'alone'
df_all.loc[(df_all['FamilySize']>=2) & (df_all['FamilySize']<=4),'FamilySize_bin'] = 'small'
df_all.loc[(df_all['FamilySize']>=5) & (df_all['FamilySize']<=7),'FamilySize_bin'] = 'mediam'


# In[ ]:


# 敬称(honorific)の加工
df_all['honorific'] = df_all['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
df_all['honorific'].replace(['Col','Dr', 'Rev'], 'Rare',inplace=True) #少数派の敬称を統合
df_all['honorific'].replace('Mlle', 'Miss',inplace=True) #Missに統合
df_all['honorific'].replace('Ms', 'Miss',inplace=True) #Missに統合


# In[ ]:


# Cabinの頭文字
df_all['Cabin_ini'] = df_all['Cabin'].map(lambda x:str(x)[0])
df_all['Cabin_ini'].replace(['G','T'], 'Rare',inplace=True) #少数派のCabin_iniを統合


# In[ ]:


# Ticket頻度
df_all.loc[:, 'TicketFreq'] = df_all.groupby(['Ticket'])['PassengerId'].transform('count')


# In[ ]:


# Fareの分割
bins = [-1, 15, 60, 600]
df_all['Fare_bin'] = pd.cut(df_all['Fare'], bins=bins)


# In[ ]:


# Cabinの頭文字
df_all['Cabin_ini'] = df_all['Cabin'].map(lambda x:str(x)[0])
df_all['Cabin_ini'].replace(['G','T'], 'Rare',inplace=True) #少数派のCabin_iniを統合


# In[ ]:


# Ageを分割
bins = [0, 15, 60, 80]
df_all['Age_bins'] = pd.cut(df_all['Age'], bins=bins)


# In[ ]:


# 不要なカラムを削除する
df_all.drop(['PassengerId', 'Name', 'Fare', 'Age', 'Cabin', 'family_name', 'SibSp', 'Parch', 'Ticket', 'FamilySize'], axis=1, inplace=True)


# In[ ]:


df_all


# In[ ]:


df_all.isnull().sum()


# In[ ]:


# Age_binの予測モデル作成
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


# カテゴリカル変数の処理しagepredのデータセットを用意
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


# labelencordingするカラムを抽出
cols = [col for col in df_all.columns
        if df_all[col].dtype != 'int']
# train_or_testをdrop
# cols.remove('train_or_test')


# In[ ]:


cols


# In[ ]:


for col in cols:
  df_all[col] = le.fit_transform(df_all[col])


# In[ ]:


df_all


# In[ ]:


# 事前に設定したフラグでデータを分離
age_train = df_all.query('train_or_test == 1')
age_test = df_all.query('train_or_test == 0')
age_train = age_train.drop(['train_or_test'], axis=1)
age_test = age_test.drop(['train_or_test'], axis=1)
age_train.describe()


# In[ ]:


# 欠損値に3が付与されている
age_train['Age_bins'].value_counts()


# In[ ]:


# Age_binが欠損値，そうでないものに切り分ける
age_train_isnull = age_train[age_train['Age_bins']==3]
age_train_full = age_train[age_train['Age_bins']!=3]
age_test_isnull = age_test[age_test['Age_bins']==3]
age_test_full = age_test[age_test['Age_bins']!=3]

# データ整理
# Ageを予測するための教師データ
age_pred_x = age_train_full.drop(['Survived'], axis=1)
age_pred_t = age_train_full['Age_bins']
# Ageを予測するためのテストデータ
age_train_isnull_d = age_train_isnull.drop(['Survived', 'Age_bins'], axis=1)
age_test_isnull_d = age_test_isnull.drop({'Survived', 'Age_bins'}, axis=1)


# In[ ]:


age_pred_x.head(10)


# In[ ]:


get_ipython().system('pip install pandas==2.0.3')


# In[ ]:


get_ipython().system('pip install numpy matplotlib seaborn altair')


# In[ ]:


get_ipython().system('pip install scikit-learn==1.0.2')


# In[ ]:


get_ipython().system('pip install pycaret')


# In[ ]:


#!apt-get update
#!apt-get install -y build-essential


# In[ ]:


from pycaret.classification import *


# In[ ]:


# データの前処理
exp = setup(
    data = age_pred_x, target='Age_bins', train_size=0.8, session_id=0
)


# In[ ]:


best_model = compare_models()


# In[ ]:


clf_pycaret = create_model(best_model, fold=5)


# In[ ]:


tuned_model = tune_model(clf_pycaret, fold=5)


# In[ ]:


predict_model(tuned_model)


# In[ ]:


params = tuned_model.get_params()
params


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


# RFECVはKNeighborsClassifierを用いて特徴量選定ができない
# 2番手のExtraTreesClassifierを用いる
clf_pycaret_2 = create_model('et', fold=5)


# In[ ]:


# チューニング
tuned_model_2 = tune_model(clf_pycaret_2, fold=5)


# In[ ]:


# パラメータを保存
params_2 = tuned_model_2.get_params()
params_2


# In[ ]:


from sklearn.feature_selection import RFECV
age_pred_x = age_pred_x.drop(['Age_bins'], axis=1)


# In[ ]:


age_train_x_val, x_test, age_train_t_val, t_test = train_test_split(age_pred_x, age_pred_t, test_size=0.4, random_state=0)


# In[ ]:


age_train_t_val.info()


# In[ ]:


age_train_x_val = age_train_x_val
age_train_x_val.info()


# In[ ]:


# 2番目に精度の高いExtraTreesClassifierで特徴量選定
estimator = ExtraTreesClassifier(**params_2)
rfecv = RFECV(estimator, step=1, cv=5, scoring='accuracy')
rfecv.fit(age_train_x_val, age_train_t_val)


# In[ ]:


# 選択された特徴量で新しいデータセットを作成
# 全部とってきとるやんけ
selected_features = age_train_x_val.columns[rfecv.get_support()]

# 選択された特徴量で新しいデータセットを作成
x_agepred_train_selected = age_train_x_val[selected_features]
x_agepred_test_selected = x_test[selected_features]
print(selected_features)


# In[ ]:


# KNeighborsClassifierはRFECVが適用できない
# KNeighborsClassifierモデルをトレーニング
knn_model = KNeighborsClassifier(**params)
knn_model.fit(x_agepred_train_selected, age_train_t_val)


# In[ ]:


# 予測する
y_pred = knn_model.predict(x_agepred_test_selected)
accuracy = accuracy_score(t_test, y_pred)
accuracy


# In[ ]:


# trainの欠損値を補完
df_age_pred_train = knn_model.predict(age_train_isnull[selected_features])
df_age_pred_test = knn_model.predict(age_test_isnull[selected_features])

age_train_isnull.loc[:, 'Age_bins'] = np.argmax(df_age_pred_train)
age_test_isnull.loc[:, 'Age_bins'] = np.argmax(df_age_pred_test)

# 補完が完了した新たなデータフレームを作成
df_train_comp = pd.concat([age_train_full, age_train_isnull], axis=0).sort_index()
df_test_comp = pd.concat([age_test_full, age_test_isnull], axis=0).sort_index()

df_train_comp.isnull().sum()


# In[ ]:


# オーバーサンプリング
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0, k_neighbors=10)

x = df_train_comp.drop('Survived', axis=1)
t = df_train_comp['Survived']
x_oversampled, t_oversampled = smote.fit_resample(x, t)

# 補完データに適用
df_train_oversampled = pd.DataFrame(x_oversampled, columns=x.columns)
df_train_oversampled['Survived'] = t_oversampled
df_train_oversampled


# In[ ]:


# データの前処理
exp = setup(
    data = df_train_oversampled, target='Survived', train_size=0.8, session_id=0
)


# In[ ]:


best_model = compare_models()


# In[ ]:


# モデルの作成
clf_pycaret = create_model(best_model, fold=10)


# In[ ]:


# ハイパーパラメータの調整
tuned_model = tune_model(clf_pycaret, fold=10)


# In[ ]:


# 8 : 2 で分割されていたテストデータへの適用
predict_model(tuned_model)


# In[ ]:


params = tuned_model.get_params()
params


# In[ ]:


# xとtの準備
x = df_train_oversampled.drop('Survived', axis = 1)
t = df_train_oversampled['Survived']
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size = 0.3, random_state = 0)


# In[ ]:


# RFECVを用いて特徴量選定
rfc = ExtraTreesClassifier(**params)
rfecv = RFECV(estimator=rfc, step=1, cv=5, scoring='accuracy')
rfecv.fit(x_train_val, t_train_val)

# 選択された特徴量の列を抽出
selected_features = x_train_val.columns[rfecv.get_support()]

# 選択された特徴量で新しいデータセットを作成
x_train_selected = x_train_val[selected_features]
x_test_selected = x_test[selected_features]

# エクストラ何ちゃらを適用
rfc.fit(x_train_selected, t_train_val)
y_pred = rfc.predict(x_test_selected)


# In[ ]:


# 予測の評価
accuracy = accuracy_score(t_test, y_pred)
accuracy


# In[ ]:


# 選択された特徴量の列を表示
selected_features


# In[ ]:


length = len(y_pred)
print(length)


# In[ ]:


# Kaggleに提出するファイルを作成
result = rfc.predict(df_test_comp[selected_features])
result


# In[ ]:


submit = pd.DataFrame(pd.read_csv('/content/test.csv')['PassengerId'])
submit['Survived'] = result
submit.to_csv('submission.csv', index=False)


# In[ ]:


submit


# In[ ]:


from google.colab import files
files.upload()  # kaggle.jsonファイルをアップロード
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle competitions submit -c titanic -f submission.csv -m "Message"')


# In[ ]:


get_ipython().system('python3 titanic.py')


# In[ ]:




