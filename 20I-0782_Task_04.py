import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('iris.csv')

# Separate features (X) and target variable (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the testing set
predictions = clf.predict(X_test)

# Print predictions and actual labels
print("Predictions:\n", predictions)
print("\nActual labels:\n", y_test.values)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nAccuracy:", accuracy)

# Save the model to a .pkl file
joblib.dump(clf, 'model.pkl')
