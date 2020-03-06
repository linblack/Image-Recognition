from sklearn.tree import DecisionTreeClassifier, export_graphviz

feature = [[1,32],[1,25],[0,26],[1,19],[0,28],[0,18],[1,17],[0,22],
           [1,29],[0,30]]
target = [1,1,0,1,0,1,1,0,1,0]
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(feature, target)
prediction = tree.predict(feature)
export_graphviz(tree, out_file='hero.dot', feature_names=['gender','age'],
                class_names=['Captain America','Iron Man'])
# print(prediction)