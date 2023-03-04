# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Reading the dataset
data = pd.read_csv('diabetes.csv')

# Splitting the dataset into train and test sets
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Making predictions on the test set
y_pred = lr.predict(X_test)

# Evaluating the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Deploying the model for prediction
def predict_diabetes(data):
    prediction = lr.predict(data)
    return prediction
