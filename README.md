# PRODIGY-3
# **Decision Tree Classifier for Predicting Customer Purchases**

- This repository contains Python code for training a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data.

## **Dataset**
- The dataset used for this project is stored in a CSV file named bank.csv. It contains information about customers, including their age, job, marital status, education, and other attributes. The target variable is whether the customer made a deposit (1) or not (0).

## **Prerequisites**
### **Make sure you have the following libraries installed:**
- pandas
- scikit-learn
- matplotlib
- seaborn
### **You can install them using pip:**
- pip install pandas scikit-learn matplotlib seaborn

## **Usage**

### **1. Clone Repository**
- git clone https://github.com/PatelHet15/decision-tree-classifier.git

### **2. Navigate to the project directory**
- cd decision-tree-classifier

### **3. Run the python script**
- python main.py

- The script will load the dataset, preprocess the data, train a decision tree classifier using GridSearchCV for hyperparameter tuning, evaluate the classifier's performance, and visualize various aspects of the model.

## **Result**

### **The output of the script includes:**
- Best parameters found during hyperparameter tuning.
- Accuracy of the best classifier on the testing set.
- Classification report for the best classifier, including precision, recall, F1-score, and support for each class.
- Confusion matrix visualizing the classifier's performance.
- ROC curve and AUC score.
- Feature importance of the best classifier.
- Percentage of customers who bought the product and who didn't buy it.

## **License**
- This project is licensed under the MIT License - see the LICENSE file for details.


  
