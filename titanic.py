import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.56)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFECV
from IPython.display import display
from IPython.display import display_html
import warnings

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

#Loading data
df_train = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\venv\titanic\train.csv')
df_test = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\venv\titanic\test.csv')
df_data = df_train.append(df_test)
warnings.filterwarnings("ignore")

# sns.countplot(df_data['Sex'], hue=df_data['Survived'])
# display(df_data[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().round(3))
# plt.show()
#
# sns.countplot(df_data['Pclass'], hue=df_data['Survived'])
# display(df_data[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().round(3))
# plt.show()

df_data['Sex_Code'] = df_data['Sex'].map({'female':1, 'male':0}).astype('int')

df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())
df_data['FareBin_4'] = pd.qcut(df_data['Fare'], 4)
df_data['FareBin_5'] = pd.qcut(df_data['Fare'], 5)
df_data['FareBin_6'] = pd.qcut(df_data['Fare'], 6)
label = LabelEncoder()
df_data['FareBin_Code_4'] = label.fit_transform(df_data['FareBin_4'])
df_data['FareBin_Code_5'] = label.fit_transform(df_data['FareBin_5'])
df_data['FareBin_Code_6'] = label.fit_transform(df_data['FareBin_6'])

df_data['Family_size'] = df_data['SibSp'] + df_data['Parch'] + 1

deplicate_ticket = []
for tk in df_data.Ticket.unique():
    tem = df_data.loc[df_data.Ticket == tk, 'Fare']
    if tem.count() > 1:
        deplicate_ticket.append(df_data.loc[df_data.Ticket == tk, ['Name','Ticket','Fare','Cabin','Family_size','Survived']])
deplicate_ticket = pd.concat(deplicate_ticket)

df_data['Connected_Survival'] = 0.5
for _, df_grp in df_data.groupby('Ticket'):
    if (len(df_grp) > 1):
        for ind, row in df_grp.iterrows():
            smax = df_grp.drop(ind)['Survived'].max()
            smin = df_grp.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 1
            elif (smin == 0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Connected_Survival'] = 0
df_data['Title'] = df_data.Name.str.extract('([A-Za-z]+)\.', expand=False)
df_data['Title'] = df_data['Title'].replace(['Capt','Col','Countess','Don','Dr','Dona','Jockheer','Major','Rev','Sir'],'Rare')
df_data['Title'] = df_data['Title'].replace(['Mlle','Ms','Mme'],'Miss')
df_data['Title'] = df_data['Title'].replace(['Lady'],'Mrs')
df_data['Title'] = df_data['Title'].map({'Mr':0, 'Rare':1, 'Master':2, 'Miss':3, 'Mrs':4})

Ti_pred = df_data.groupby('Title')['Age'].median().values
df_data['Ti_Age'] = df_data['Age']
for i in range(0,5):
    df_data.loc[(df_data.Age.isnull())&(df_data.Title == i),'Ti_Age'] = Ti_pred[i]
df_data['Ti_Age'] = df_data['Ti_Age'].astype('int')
df_data['Ti_Minor'] = ((df_data['Ti_Age']) < 16.0) * 1

# print(df_data.groupby('Connected_Survival')[['Survived']].mean().round(3))

# print(deplicate_ticket.head(14))
# df_fri = deplicate_ticket.loc[(deplicate_ticket.Family_size == 1)&(deplicate_ticket.Survived.notnull())].head(7)
# df_fam = deplicate_ticket.loc[(deplicate_ticket.Family_size > 1)&(deplicate_ticket.Survived.notnull())].head(7)
# display(df_fri,df_fam)

df_train = df_data[:len(df_train)]
df_test = df_data[len(df_train):]

X = df_train.drop(labels=['Survived','PassengerId'],axis=1)
Y = df_train['Survived']

#Show Baseline
# Base = ['Sex_Code','Pclass']
# Compare = ['Sex_Code','Pclass','FareBin_Code_4','FareBin_Code_5','FareBin_Code_6']
# b4, b5, b6 = ['Sex_Code','Pclass','FareBin_Code_4'],['Sex_Code','Pclass','FareBin_Code_5'],['Sex_Code','Pclass','FareBin_Code_6']
minor = ['Sex_Code','Pclass','FareBin_Code_5','Connected_Survival','Ti_Minor']
# # b4_Model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)
# # b4_Model.fit(X[b4],Y)
minor_Model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)
minor_Model.fit(X[minor],Y)
# b6_Model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)
# b6_Model.fit(X[b6],Y)
# print('b4 oob score :%.5f' %(b4_Model.oob_score_),' LB_Public : 0.7790')
print('connect oob score :%.5f' %(minor_Model.oob_score_))
# print('b6 oob score :%.5f' %(b6_Model.oob_score_),' LB_Public : 0.77033')

# #Submit
X_Submit = df_test.drop(labels=['PassengerId'], axis=1)
minor_pred = minor_Model.predict(X_Submit[minor])
submit = pd.DataFrame({'PassengerId':df_test['PassengerId'],
                       'Survived':minor_pred.astype(int)})
submit.to_csv('submit_minor.csv',index=False)