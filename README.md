# Credit_Risk_Analysis

The goal of this Jupyter notebook is to use supervised machine learning model by following a basic pattern: model-fit-predict. In this three-stage pattern, I presented a machine learning algorithm with data (the model stage), and the algorithm learns from this data (the fit stage) to form a predictive model (the predict stage).<br />
<br />
Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. Various techniques are used in this analysis to train and evaluate models with imbalanced classes that can identify the creditworthiness of borrowers.  A dataset of historical lending activity from a peer-to-peer lending services company is analyzed to identify the creditworthiness of borrowers.

---

## Required Installs

### Language: Python 3.9.12

### Libraries used:

[NumPy](https://pandas.pydata.org/pandas-docs/stable/index.html) - The fundamental package for scientific computing with Python

[Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) - For the creation and manipulation of Data Frames

[Jupyter Labs](https://jupyter.org/) - An ipython kernel for interactive computing in python

[OS](https://docs.python.org/3/library/os.html) - Miscellaneous operating system interface

[Pathlib](https://docs.python.org/3/library/pathlib.html) - Object-oriented filesystem paths

[Sklearn](https://scikit-learn.org/stable/index.html) - Scikit-learn: Machine Learning library, Simple and efficient tools for predictive data analysis

[Imblearn](https://pypi.org/project/imblearn/) - Imbalanced Learn: Over and Under-sampling library

---

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

---

## Data Analysis
<br />
When analyzing data, it's often not enough to train and then use a machine learning model for making predictions. We also need to know how well the model performs at its prediction task.  We also want to know the percentage of predictions that the model gets right and how well it predicts each outcome. We can use the following metrics to give us additional insight into the model’s performance: Accuracy, Precision, and Recall.

First, the **accuracy** of this analysis is 4 percent higher for the resampled data than the original model (0.99 vs 0.95), meaning that the model using resampled data was better at detecting true positives and true negatives.<br />
<br />
The **precision** for the "unhealthy" class is essentially equal in both data models with the original data (0.85) and the resampled data (0.84). This means that we can be about 85% confident that both models made positive preditions.<br />
![Original Data](https://github.com/stipptracie/Credit_Risk_Analysis/blob/main/Resources/original_classification_report.png)<br />
<br />
In terms of the **recall**, however, the "unhealthy" metric using resampled data was much better (0.99 vs 0.91). Meaning that the resampled data correctly clasified a higher percentage of the truly defaulting borrowers.<br />
![Resampled Data](https://github.com/stipptracie/Credit_Risk_Analysis/blob/main/Resources/resampled_classification_report.png)<br />
<br />
All in, the model using **resampled data** was much better at detecting borrowers who are likely to default than the model generated using the original, imbalanced dataset.<br />

---

### Contributors

Created by Tracie Stipp
>
> email: stipptracie@gmail.com |
> [GitHub](https://github.com/stipptracie) |
> [LinkedIn](https://www.linkedin.com/in/tracie-stipp-0719691b/)


