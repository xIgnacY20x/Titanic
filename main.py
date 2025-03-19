import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

if __name__ == '__main__':

    df = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    df_test['Title'] = df_test['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Mlle', 'Mme', 'Don', 'Dona', 'Lady', 'Countess', 'Capt', 'Sir', 'Jonkheer']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df_test['Title'] = df_test['Title'].replace(rare_titles, 'Rare')

    df['FamilySize'] = df['SibSp'] + df['Parch']
    df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch']

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])
    df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=[0, 12, 18, 35, 60, 80], labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior'])

    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    df_test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    X = df.drop(columns=['Survived', 'PassengerId'])
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = ['Age', 'Fare', 'FamilySize']
    cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup']

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42))
    ])
    '''model2 = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                     subsample=0.8, colsample_bytree=0.8,reg_lambda=10, random_state=42))
    ])
    model2.fit(X_train, y_train)'''

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    print('Skuteczność modelu:', acc)

    X_test = df_test.drop(columns=['PassengerId'])

    y_test_pred = model.predict(X_test)

    submission = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_test_pred})
    submission.to_csv('Titanic_predictions.csv', index=False)

