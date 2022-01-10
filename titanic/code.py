from google.colab import files

uploaded = files.upload()

import pandas as pd

trainData = pd.read_csv('train.csv')

testData = pd.read_csv('test.csv')
trainData.head()

trainData.shape
trainData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
trainData.head()
trainData.isnull().sum()
trainData['Age'].describe()
trainData['Age'].fillna(trainData['Age'].mean(), inplace=True)
trainData.isnull().sum()
dum1 = pd.get_dummies(trainData['Sex'], drop_first=True)
trainData = pd.concat([trainData, dum1], axis=1)
trainData.head()
trainData.drop(['Sex'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler
slr = StandardScaler()
scaleList = ['Age', 'Fare']
trainData[scaleList] = slr.fit_transform(trainData[scaleList])

trainData.head()

X = trainData.drop(['Survived'], axis=1)
y = trainData['Survived']

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    'DecisionTreeClassifier' : {
      'model' : DecisionTreeClassifier(),
      'param' : {
          'criterion' : ['gini', 'entropy']
      }
    },

    'KNeighborsClassifier' : {
      'model' : KNeighborsClassifier(),
      'param' : {
          'n_neighbors' : [5, 10, 15, 20, 25]
      }
    },

    'SVC' : {
      'model' : SVC(),
      'param' : {
          'kernel' : ['rbf', 'linear', 'sigmoid'],
          'C' : [0.1, 1, 10, 100]
      }
    },
}

accu = []

for model, p in models.items():
  modelSelect = GridSearchCV(estimator=p['model'], param_grid=p['param'], cv=5, return_train_score=False)
  modelSelect.fit(X, y)
  accu.append({
      'model' : model,
      'best_score' : modelSelect.best_score_,
      'best_params' : modelSelect.best_params_
  })

df_score = pd.DataFrame(accu, columns=['model', 'best_score', 'best_params'])
df_score

svcModel = SVC(C=100, kernel='rbf')

svcModel.fit(X, y)

test1 = testData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

test1.isnull().sum()

test1['Age'].fillna(test1['Age'].mean(), inplace=True)
test1['Fare'].fillna(test1['Fare'].mean(), inplace=True)

dum2 = pd.get_dummies(test1['Sex'], drop_first=True)
test1 = pd.concat([test1, dum2], axis=1)
test1.drop(['Sex'], axis=1, inplace=True)

test1[scaleList] = slr.fit_transform(test1[scaleList])

test1.head()

"""Now all we need to do is predict."""

predictValue = svcModel.predict(test1)

answer = pd.DataFrame({
    'PassengerId' : testData['PassengerId'],
    'Survived' : predictValue
})

answer.to_csv('solution.csv', index=False)
