# Module 12 Report Template

## Overview of the Analysis

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. Various techniques are used in this analysis to train and evaluate models with imbalanced classes that can identify the creditworthiness of borrowers.  A dataset of historical lending activity from a peer-to-peer lending services company is analyzed to identify the creditworthiness of borrowers.<br />
<br />
In the given dataset, a value of 0 in the “loan_status” column means that the loan/borrower is healthy or creditworthy. A value of 1 means that the loan/borrower is unhealthy, or has a high risk of defaulting.<br />
<br />
In this analysis, I used a supervised learning model by following a basic pattern: model-fit-predict. In this three-stage pattern, I presented a machine learning algorithm with data (the model stage), and the algorithm learns from this data (the fit stage) to form a predictive model (the predict stage).<br />
<br />

**Methods used**<br />
* Split the Data into Training and Testing Sets<br />
    `from sklearn.model_selection import train_test_split`<br />
    `X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)`<br />
* Create a Logistic Regression Model with the Original Data<br />
    `from sklearn.linear_model import LogisticRegression`<br />
    `model = LogisticRegression(random_state=1)`<br />
    `lr_orginal_model = model.fit(X_train, y_train)`<br />
* Predict a Logistic Regression Model with Resampled Training Data<br />
    `y_original_pred = lr_orginal_model.predict(X_test)`<br />
* Evaluate the original and resampled data model’s performance by doing the following:<br />
    * Calculate the accuracy score of the model.<br />
            `baso = balanced_accuracy_score(y_test, y_original_pred)`<br />
    * Generate a confusion matrix.<br />
            `confusion_matrix(y_test, y_original_pred)`<br />
    * Print the classification report.<br />
            `print(classification_report_imbalanced(y_test, y_original_pred))`<br />
<br />

## Results

### **Machine Learning Model 1:** <br />
  * **Description of Model 1 Accuracy, Precision, and Recall scores**<br />
    * Accuracy: 0.9520479254722232<br />
    * Precision: 0.85<br />
    * Recall: 0.91<br />
<br />

### **Machine Learning Model 2:**
  * **Description of Model 2 Accuracy, Precision, and Recall scores**
    * Accuracy: 0.9936781215845847
    * Precision: 0.84
    * Recall: 0.99
<br />

## Summary

A classification algorithm results in two or more outcomes. Classifying unhealthy loans, for example, results in two outcomes: a loan borrower is either at risk for defaulting (unhealthy) or not. We can categorize these two predictions according to a confusion matrix.<br />
<br />
Note that the first column in the confusion matrix contains the positive predictions. That is, it separates all the predictions that the model made for the positive class into whether they were accurate or not. (In our example, the positive class is the “healthy” class.) The second column contains the negative predictions. That is, it separates all the predictions that the model made for the negative class. (In our example, the negative class is the “unhealthy” class.)<br />
<br />
To further explain, any prediction falls into one of two categories: true or false. In the context of credit worthiness, a true prediction means that the model categorized the loan/borrower as healthy. A false prediction means that the model categorized the loan/borrower as unhealthy.<br />
<br />

* If the model predicted a loan/borrower as healthy, and the loan/borrower really was healthy, we call that prediction a **true positive (TP)**.

* If the model predicted a loan/borrower as healthy, but the loan/borrower was not healthy, we call that prediction a **false positive (FP)**.

* If the model predicted a loan/borrower as not healthy, but the loan/borrower really was healthy, we call that prediction a **false negative (FN)**.

* If the model predicted a loan/borrower as not healthy, and the loan/borrower really was not healthy, we call that prediction a **true negative (TN)**.
<br />

**Definition of terms and formulas:**<br />
* **Accuracy:** measures how often the model was correct.<br />
    * *Formula: accuracy = (TPs + TNs) ÷ (TPs + TNs + FPs + FNs)* <br />
        * It does so by calculating the ratio of the number of correct predictions to the total number of outcomes.<br />
* **Precision:** measures how confident we are that the model correctly made the positive predictions (also known as the positive predictive value (PPV)).<br />
    * *Formula: precision = TPs ÷ (TPs + FPs)*<br />
        * We get the precision by dividing the number of TPs by the number of all the positives. (The latter is the sum of the TPs and the FPs.)<br />
* **Recall:** measures the number of actually fraudulent transactions that the model correctly classified as fraudulent.<br />
    * *Formula: Recall = TPs / (TPs + FNs)*<br />
        * To get the recall, we start with the number of TPs — that is, the number of times that the model correctly predicted a fraudulent transaction. Then compare this number to the total number of actually fraudulent transactions — including the ones that the model missed (that is, the FNs).<br />
<br />

**Analysis of Results**<br />
<br />
First, the accuracy score is a small amount higher for the resampled data (0.99 vs 0.95), meaning that the model using resampled data was better at detecting true positives and true negatives.<br />
<br />
The precision for the minority class is higher with the original data (0.55) versus the resampled data (0.32) meaning that the original data was better at detecting the users that were actually going to default. Original - 0.85; Resampled - 0.84
![Original Data]("../Resources/original_classification_report.png")
<br />
In terms of the recall, however, the minority class metric using resampled data was much better (0.82 vs 0.15). Meaning that the resampled data correctly clasified a higher percentage of the truly defaulting borrowers.  Original - 0.91; Resampled - 0.99
![Resampled Data]("../Resources/resampled_classification_report.png")
<br />
All in, the model using resampled data was much better at detecting borrowers who are likely to default than the model generated using the original, imbalanced dataset.<br />


