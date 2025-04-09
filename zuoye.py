

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

data = "E:\code\codinghomework\iris.csv"

iris_local = pd.read_csv(data, usecols=[0, 1, 2, 3, 4])
iris_local = iris_local.dropna()

'''iris_local.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()
iris_local.hist()
plt.show()
iris_local.plot()
plt.show()'''
#画图

from sklearn.feature_selection import SelectKBest, f_classif

X = iris_local.drop('target', axis=1)  # 特征
y = iris_local['target']               # 目标变量

#标记特征

selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("Selected Features:", selected_features)
#特征选择
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
print("保留方差比例:", pca.explained_variance_ratio_.sum())

from sklearn.decomposition import PCA
pca = PCA(n_components=4).fit(X)
print("保留方差比例:", pca.explained_variance_ratio_.sum())
#查看准确度

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target
features = iris_dataset.feature_names

# 原始数据性能
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
svc = SVC(kernel='linear').fit(X_train, y_train)
orig_acc = accuracy_score(y_test, svc.predict(X_test))

# 降维后性能
X_train_red, X_test_red = X_train[:, [2,3]], X_test[:, [2,3]]
svc_red = SVC(kernel='linear').fit(X_train_red, y_train)
red_acc = accuracy_score(y_test, svc_red.predict(X_test_red))


print(f"4准确率：{orig_acc:.4f}")
print(f"2准确率：{red_acc:.4f}")
print(f"准确率变化：{(red_acc-orig_acc):.4f}")



















































'''iris = load_iris()
X = iris.data  # 特征 (150 samples, 4 features)
y = iris.target  # 标签 (3类别)
feature_names = iris.feature_names
target_names = iris.target_names


df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
print(df.head())
print(iris["DESCR"][:193])
print("Target names: {}".format(iris["target_names"]))
print("Feature names: {}".format(iris['feature_names']))
print("Type of data: {}".format(type(iris['data'])))
print("shape of data: {}".format(iris['data'].shape))
print("First five rows of data:\n{}".format(iris['data'][:5]))
print("target:\n{}".format(iris['target']))'''
