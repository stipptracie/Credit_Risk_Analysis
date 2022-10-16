# Module 12 Report Template

## Overview of the Analysis

* Explain the purpose of the analysis.
Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. I will use various techniques to train and evaluate models with imbalanced classes that can identify the creditworthiness of borrowers.

* Explain what financial information the data was on, and what you needed to predict.
I used a dataset of historical lending activity from a peer-to-peer lending services company to identify the creditworthiness of borrowers.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

* Describe the stages of the machine learning process you went through as part of this analysis.
In this analysis, I used a supervised learning model by following a basic pattern: model-fit-predict. In this three-stage pattern, I presented a machine learning algorithm with data (the model stage), and the algorithm learns from this data (the fit stage) to form a predictive model (the predict stage).

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
I used the following steps:
    * Split the Data into Training and Testing Sets
    
        '''
        # Import the train_test_learn module
        from sklearn.model_selection import train_test_split
        # Split the data using train_test_split
        # Assign a random_state of 1 to the function
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        '''
        
    * Create a Logistic Regression Model with the Original Data
    
        '''
        # Import the LogisticRegression module from SKLearn
        from sklearn.linear_model import LogisticRegression
        # Instantiate the Logistic Regression model
        # Assign a random_state parameter of 1 to the model
        model = LogisticRegression(random_state=1)
        # Fit the model using training data
        lr_orginal_model = model.fit(X_train, y_train)
        '''
        
    * Predict a Logistic Regression Model with Resampled Training Data
        
        '''
        # Make a prediction using the testing data
        y_original_pred = lr_orginal_model.predict(X_test)
        '''

I evaluated the original and resampled data model’s performance by doing the following:
    * Calculate the accuracy score of the model.
    
        '''
        # Print the balanced_accuracy score of the model
        baso = balanced_accuracy_score(y_test, y_original_pred)
        '''
        
    * Generate a confusion matrix.
    
        '''
        # Generate a confusion matrix for the model
        confusion_matrix(y_test, y_original_pred)
        '''
        
    * Print the classification report.
    
        '''
        # Print the classification report for the model
        print(classification_report_imbalanced(y_test, y_original_pred))
        '''
        
## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

A classification algorithm results in two or more outcomes. Classifying unhealthy loans, for example, results in two outcomes: a loan borrower is either at risk for defaulting (unhealthy) or not. We can categorize these two predictions according to a confusion matrix.

Note that the first column in the confusion matrix contains the positive predictions. That is, it separates all the predictions that the model made for the positive class into whether they were accurate or not. (In our example, the positive class is the “healthy” class.) The second column contains the negative predictions. That is, it separates all the predictions that the model made for the negative class. (In our example, the negative class is the “unhealthy” class.)

To further explain, any prediction falls into one of two categories: true or false. In the context of credit worthiness, a true prediction means that the model categorized the loan/borrower as healthy. A false prediction means that the model categorized the loan/borrower as unhealthy.

If the model predicted a loan/borrower as healthy, and the loan/borrower really was healthy, we call that prediction a true positive (TP).

If the model predicted a loan/borrower as healthy, but the loan/borrower was not healthy, we call that prediction a false positive (FP).

If the model predicted a loan/borrower as not healthy, but the loan/borrower really was healthy, we call that prediction a false negative (FN).

If the model predicted a loan/borrower as not healthy, and the loan/borrower really was not healthy, we call that prediction a true negative (TN).

Definition of terms and formulas:
    * Accuracy: measures how often the model was correct. 
          Formula: accuracy = (TPs + TNs) ÷ (TPs + TNs + FPs + FNs) 
              It does so by calculating the ratio of the number of correct predictions to the total number of outcomes.
    * Precision: measures how confident we are that the model correctly made the positive predictions (also known as the positive predictive value (PPV)). 
          Formula: precision = TPs ÷ (TPs + FPs)
              We get the precision by dividing the number of TPs by the number of all the positives. (The latter is the sum of the TPs and the FPs.)
    * Recall: measures the number of actually fraudulent transactions that the model correctly classified as fraudulent. 
          Formula: Recall = TPs / (TPs + FNs) 
              To get the recall, we start with the number of TPs — that is, the number of times that the model correctly predicted a fraudulent transaction. Then compare this number to the total number of actually fraudulent transactions — including the ones that the model missed (that is, the FNs).
              
* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
        * Accuracy: 0.9520479254722232
        * Precision: 0.85
        * Recall: 0.91

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
        * Accuracy: 0.9936781215845847
        * Precision: 0.84
        * Recall: 0.99

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

???? The results regarding accuracy of the minority class are actually mixed when comparing the classifiction reports generated from the predictions with the original data versus the predictions with the resampled data.

First, the accuracy score is a small amount higher for the resampled data (0.99 vs 0.95), meaning that the model using resampled data was better at detecting true positives and true negatives.   

The precision for the minority class is higher with the original data (0.55) versus the resampled data (0.32) meaning that the original data was better at detecting the users that were actually going to default. Original - 0.85; Resampled - 0.84
![][""]

In terms of the recall, however, the minority class metric using resampled data was much better (0.82 vs 0.15). Meaning that the resampled data correctly clasified a higher percentage of the truly defaulting borrowers.  Original - 0.91; Resampled - 0.99
![][""]

f1 - Original - 0.88, Resampled - 0.91

All in, the model using resampled data was much better at detecting borrowers who are likely to default that the model generated using the original, imbalanced dataset.

If you do not recommend any of the models, please justify your reasoning.
