from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

wine_data = datasets.load_wine()
train_feature, test_feature, train_target, test_target = train_test_split(
    wine_data.data, wine_data.target, test_size=0.3)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
tree.fit(train_feature, train_target)

forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                max_depth=4)
forest.fit(train_feature, train_target)
accuracy_tree = tree.score(test_feature, test_target)
accuracy_forest = forest.score(test_feature, test_target)
print('accuracy_tree:',accuracy_tree)
print('accuracy_forest:',accuracy_forest)