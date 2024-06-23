1st Model: XGBClassifier:

The bar chart displays the proportion of delayed flights by day of the month, providing insight into the busiest days.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/4308526f-23be-4a14-a929-339d986e1605)

The bar chart displays the number of delayed flights by airline, highlighting the airlines with the most and least delays.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/45daadf7-30ec-4725-8c4c-d567689cde74)

The graphs below display the distributions and trends of all variables in the dataset.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/7c33e5c0-c4c8-4464-a9c5-6c9a379ff4bd)

The correlation matrix shows the relationship between different variables in a dataset. 
It displays correlation coefficients, which quantify the strength and direction of the linear relationship between pairs of variables. 
There are strong relationships between several variables: Flight Status and Departure Delay (0.71), Speed and Airtime (0.58), and Distance and Speed (0.68).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/fec406be-6605-4714-baf4-a4cf87d44041)

The XGBClassifier model achieved quite high performance scores, including an accuracy of 0.94 and an F1 score of 0.81. 
This high performance is further reflected in the confusion matrix below, which shows the True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/33ee2b78-68f4-4289-b834-c8dba14f2f44)

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/dd64c538-93fb-461a-8832-f3f8284db542)

The Permutation Feature Importance graph below shows how each feature contributes to the model's performance. 
Flight Dates, Departure Delay, and Speed are the most significant contributors, while Origin and Departure Cities contribute the least.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/c3e62427-2308-4c10-9a5d-c18f4772b5db)


2nd Model: Decision Tree Classifier:

The Confusion Matrix below shows slightly lower performance than Confusion Matrix of 1st model with the following values: True Negatives (TN): 153,982, False Positives (FP): 15,989, False Negatives (FN): 8,818, and True Positives (TP): 30,926.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/56bf2bbb-29f0-4442-8ac6-8ac56a68bdb0)

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/94a9bf07-1d45-47db-9fa5-b0786263ab26)

Departure Delay is the biggest contributor in the 2nd model, whereas it was the 4th largest contributor in the 1st model.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/b7e77424-449e-45ec-84b3-7fd7fb9486a4)
