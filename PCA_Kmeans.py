from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#PCA
iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=42)   #stratify=y 按Y比例分配
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, Y_train)
preds = clf.predict_proba(X_test)   #預測各類別的機率，總和為1
print('Accuracy:',accuracy_score(Y_test, preds.argmax(axis=1))) #argmax回傳最大值index，透過accuracy_score與Y_test比較，計算accuracy

pca = decomposition.PCA(n_components=2) #分成2類
X_centerrd = X-X.mean(axis=0)
pca.fit(X_centerrd)
X_pca = pca.transform(X_centerrd)   #使用PCA降維，資料分成0,1,2三組，各50筆

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, stratify=Y, random_state=42)
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, Y_train)
preds = clf.predict_proba(X_test)
print('Accuracy:',accuracy_score(Y_test, preds.argmax(axis=1)))
# plt.plot(X_pca[Y==0,0], X_pca[Y==0,1], 'bo', label='Setosa')
# plt.plot(X_pca[Y==1,0], X_pca[Y==1,1], 'go', label='Versicolour')
# plt.plot(X_pca[Y==2,0], X_pca[Y==2,1], 'ro', label='Virginica')
# plt.legend()
# plt.show()


#K-means
clf = KMeans(n_clusters=3)
clf.fit(X_pca)
preds = clf.labels_ #回傳分類結果[0 0 2 1...]
plt.scatter(X_pca[:,0], X_pca[:,1], c=preds)   #c=preds 依照分類結果塗色
plt.show()