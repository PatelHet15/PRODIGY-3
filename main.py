import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# Load the data from the CSV file
data = pd.read_csv('bank.csv')

# Drop any rows with missing values
data.dropna(inplace=True)

# Convert categorical variables to numerical labels
label_encoder = LabelEncoder()
data['job'] = label_encoder.fit_transform(data['job'])
data['marital'] = label_encoder.fit_transform(data['marital'])
data['education'] = label_encoder.fit_transform(data['education'])
data['default'] = label_encoder.fit_transform(data['default'])
data['housing'] = label_encoder.fit_transform(data['housing'])
data['loan'] = label_encoder.fit_transform(data['loan'])
data['contact'] = label_encoder.fit_transform(data['contact'])
data['month'] = label_encoder.fit_transform(data['month'])
data['poutcome'] = label_encoder.fit_transform(data['poutcome'])
data['deposit'] = label_encoder.fit_transform(data['deposit'])

# Select features and target variable
X = data.drop('deposit', axis=1)
y = data['deposit']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5)

# Perform grid search to find the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best model
best_clf = grid_search.best_estimator_

# Make predictions on the testing set using the best model
y_pred_best = best_clf.predict(X_test)

# Evaluate the best classifier
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Classifier Accuracy:", accuracy_best)

# Display classification report for the best classifier
print("Classification Report for Best Classifier:")
print(classification_report(y_test, y_pred_best))

# Visualize the decision tree of the best classifier
plt.figure(figsize=(100,100))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the best classifier to a file
joblib.dump(best_clf, 'best_classifier.pkl')

# Cross-validation scoring
cv_scores = cross_val_score(best_clf, X, y, cv=5)
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Average Accuracy Score:", cv_scores.mean())

# Calculate ROC curve and AUC score
y_scores = best_clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Feature importance visualization
feature_importances = best_clf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance of the Best Classifier')
plt.show()

# Calculate the percentage of customers who bought the product
buy_percentage = (y_pred_best.sum() / len(y_pred_best)) * 100

# Calculate the percentage of customers who didn't buy the product
no_buy_percentage = 100 - buy_percentage

print("Percentage of customers who bought the product: {:.2f}%".format(buy_percentage))
print("Percentage of customers who didn't buy the product: {:.2f}%".format(no_buy_percentage))