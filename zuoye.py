'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris

data = "E:\code\codinghomework\iris.csv"

iris_local = pd.read_csv(data, usecols=[0, 1, 2, 3, 4])
iris_local = iris_local.dropna()

iris_local.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()
iris_local.hist()
plt.show()
iris_local.plot()
plt.show()
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

'''
#以下是LDA
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()

iris_X, iris_Y = iris.data, iris.target

label_dict = {i:k for i,k in enumerate(iris.target_names)}

mean_vectors = []
for cls in [0, 1, 2]:
    class_mean_vector = np.mean(iris_X[iris_Y==cls],axis=0)
    mean_vectors.append(class_mean_vector)
    print(label_dict[cls], class_mean_vector)

S_W = np.zeros((4,4))
for cls, mv in zip(range(3), mean_vectors):
    S_cls = np.zeros((4,4))
    for row in iris_X[iris_Y == cls]:
        row, mv = row.reshape(4,1), mv.reshape(4,1)
        S_cls += (row-mv).dot((row-mv).T)
    S_W += S_cls
print(S_W)

all_mean = np.mean(iris_X, axis=0).reshape(4,1)
S_B = np.zeros((4,4))
for cls, mv in zip(range(3), mean_vectors):
    mv = mv.reshape(4,1)
    S_B += len(iris_X[iris_Y == cls])*(mv-all_mean).dot((mv-all_mean).T)
print(S_B)

eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(S_W), S_B))
eig_vals = eig_vals.real
eig_vecs = eig_vecs.real
for i in range(len(eig_vals)):
    eig_vec_cls = eig_vecs[:,i]
    print('特征向量 {}: {}'.format(i+1, eig_vec_cls))
    print('特征值 {}: {}'.format(i+1, eig_vals[i]))

linear_discriminants = eig_vecs.T[:2]

def plot(X, y, title, x_label, y_label):
  ax = plt.subplot(111)
  for label,marker,color in zip(range(3),('^', 's', 'o'),('blue', 'red', 'green')):
    plt.scatter(x=X[:,0].real[y == label], y=X[:,1].real[y == label], color=color, alpha=0.5, label=label_dict[label] )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
  leg = plt.legend(loc='upper right', fancybox=True)
  leg.get_frame().set_alpha(0.5)
  plt.title(title)

lda_iris_projection = np.dot(iris_X, linear_discriminants.T)
lda_iris_projection[:5,]

plot(lda_iris_projection, iris_Y, 'LDA Projection', 'LDA1', 'LDA2')

plt.show()

























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
