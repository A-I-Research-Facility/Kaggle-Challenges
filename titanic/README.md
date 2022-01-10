💢 We will be making this program using 'pandas' and 'scikit learn' libraries.

First step is to upload the datasets to the server.
We can do that as below :

      from google.colab import files
      uploaded = files.upload()
      
***

💢 Now we begin our program.

      import pandas as pd

      trainData = pd.read_csv('train.csv')

      testData = pd.read_csv('test.csv')

***

💢 Now, we will have a look at the first 5 record of the training data. This is done in order to see and analyze the types of categories present in the dataset.

      trainData.head()

We have verified that our dataset is now loaded and is ready to be used.

***

💢 Next step is to analyze the shape of the dataset. This is done to verify the number of records and columns in dataset.

      trainData.shape

This means that our dataset has a total of 891 records and 12 columns.

Now, we should drop those columns from our dataset which don't play a role in determining the 'ground truth'.

***

💢 Notice that in the database, 'Ticket', 'Fare', 'Cabin' are interrelated. Hence we only consider one out of the three. It is easier to consider 'Fare' as it is numerical in value.

      trainData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

      trainData.head()

***

💢 Now we should check whether or not there are any missing values in our dataset. If dataset is missing values, we need to fill in the empty spaces.

      trainData.isnull().sum()

As we can see, dataset is missing values in AGE column. We will now get the stats of AGE column to decide and choose which vlue to fill in.

      trainData['Age'].describe()

Here we can see the stats of AGE column. I have decided to fill the empty spaces with average value, i.e., the mean.

      trainData['Age'].fillna(trainData['Age'].mean(), inplace=True)

      trainData.isnull().sum()

***

💢 Now all the values have been filled. Next, we need to change any categorical data into numerical value. Since SEX has string values, we will change it into numerical values.

There are several methods to do this. We can use LabelEncoder from scikitLearn library, or we can also use dummy encoding which is faster in the given scenario.

Hence we will use dummy encoding.

      dum1 = pd.get_dummies(trainData['Sex'], drop_first=True)

      trainData = pd.concat([trainData, dum1], axis=1)

      trainData.head()

Since now we have dummy column attached to dataset, we dont need SEX column from now on.

      trainData.drop(['Sex'], axis=1, inplace=True)

***

💢 Next step is to scale the values of dataset. Since AGE and FARE have very different values as compared to rest of the features, we need to scale them in order to get efficient working of our predictive model.

      from sklearn.preprocessing import StandardScaler

      slr = StandardScaler()

      scaleList = ['Age', 'Fare']
      trainData[scaleList] = slr.fit_transform(trainData[scaleList])

      trainData.head()

Now data preprocessing is over. 

***

💢 Next, we will split the dataset into X and Y variables.
Since SURVIVED is our target variable, we will put it as Y, and rest others as X.

      X = trainData.drop(['Survived'], axis=1)
      y = trainData['Survived']

***

💢 There are various kinds of models that we can use for prediction. We dont know which one will be best at this point.

Therefore, we will use grid search to select model among : 
1) Decision tree
2) KNN
3) SVM

To use grid search, we first create a dictionary with all models and their parameters to calculate accuracy.

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

***

💢 Now we have the dictionary, so we need to calculate the accuracy for each model.

To see the accuracy of each model, we also need to create a new dataframe.

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

***

💢 As we can see, SVM provides the best accuracy of about 81%, so we will use that model.

      svcModel = SVC(C=100, kernel='rbf')

      svcModel.fit(X, y)

***

💢 Now we will apply all preprocessing activities on the test dataset.

      test1 = testData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)

      test1.isnull().sum()

      test1['Age'].fillna(test1['Age'].mean(), inplace=True)
      test1['Fare'].fillna(test1['Fare'].mean(), inplace=True)

      dum2 = pd.get_dummies(test1['Sex'], drop_first=True)
      test1 = pd.concat([test1, dum2], axis=1)
      test1.drop(['Sex'], axis=1, inplace=True)

      test1[scaleList] = slr.fit_transform(test1[scaleList])

      test1.head()

***

💢 Now all we need to do is predict.

      predictValue = svcModel.predict(test1)

      answer = pd.DataFrame({
          'PassengerId' : testData['PassengerId'],
          'Survived' : predictValue
      })

***

💢 To get a csv file of the predicted value, run the following code.

      answer.to_csv('solution.csv', index=False)
