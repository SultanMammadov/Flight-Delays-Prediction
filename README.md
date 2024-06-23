**Introduction:**

Flight delays are a significant concern for air transportation worldwide due to the financial losses they cause. 
In 2022, nearly a quarter of US flights were delayed, causing significant financial impact. The aviation industry faced costs estimated between $30 billion and $34 billion due to these delays and cancellations. 
These delays inconvenience both airlines and passengers, leading to increased expenses for food and lodging, stress, and distrust towards airlines. 
Airlines also face extra costs such as crew expenses, aircraft repositioning, and increased fuel consumption, which damage their reputation and reduce demand.

Delays occur for various reasons, including air congestion, weather conditions, mechanical issues, boarding difficulties, and airlines' inability to manage demand.

Can passengers avoid delayed flights or predict delays before boarding? Using Machine Learning (ML) algorithms, it is possible to predict flight delays to some extent. Different algorithms have varying accuracy and depend on the data provided.

In this project, I explored and compared two models to predict flight delays and recommended the one with higher performance. I also examined the most contributing features in both models and observed how they change from one model to another.

**Objective:**
The goal of this project is straightforward: developing a model that predicts flight delays before they appear on the departure boards.

**Data Collection:**
The dataset, sourced from Kaggle, includes multi-year data spanning from 2009 to 2023.


**1st Model: XGBClassifier:**

The bar chart displays the proportion of delayed flights by day of the month, providing insight into the busiest days.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/4308526f-23be-4a14-a929-339d986e1605)

Figure_1. "Proportion of Delayed Flights by Day of Month"

The bar chart displays the number of delayed flights by airline, highlighting the airlines with the most and least delays.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/45daadf7-30ec-4725-8c4c-d567689cde74)

Figure_2. "The number of Delayed Flights by airline"

The graphs below display the distributions and trends of all variables in the dataset.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/7c33e5c0-c4c8-4464-a9c5-6c9a379ff4bd)

Figure_3. "Distributions and trends of all variables"

The correlation matrix shows the relationship between different variables in a dataset. 
It displays correlation coefficients, which quantify the strength and direction of the linear relationship between pairs of variables. 
There are strong relationships between several variables: Flight Status and Departure Delay (0.71), Speed and Airtime (0.58), and Distance and Speed (0.68).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/fec406be-6605-4714-baf4-a4cf87d44041)

Figure_4. "Correlation Matrix of all variables"

The XGBClassifier model achieved quite high performance scores, including an accuracy of 0.94 and an F1 score of 0.81. 
This high performance is further reflected in the confusion matrix below, which shows the True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/33ee2b78-68f4-4289-b834-c8dba14f2f44)

Figure_5. "Confusion Matrix"

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/dd64c538-93fb-461a-8832-f3f8284db542)

Figure_6. "ROC (Receiver Operating Characteristic) for the result of XGBClassifier"

The Permutation Feature Importance graph below shows how each feature contributes to the model's performance. 
Flight Dates, Departure Delay, and Speed are the most significant contributors, while Origin and Departure Cities contribute the least.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/c3e62427-2308-4c10-9a5d-c18f4772b5db)

Figure_7. "Permutation Feature Importance for the of XGBClassifier"

**2nd Model: Decision Tree Classifier:**

The Confusion Matrix below shows slightly lower performance than Confusion Matrix of 1st model with the following values: True Negatives (TN): 153,982, False Positives (FP): 15,989, False Negatives (FN): 8,818, and True Positives (TP): 30,926.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/56bf2bbb-29f0-4442-8ac6-8ac56a68bdb0)

Figure_8. "Confusion Matrix"

The ROC (Receiver Operating Characteristic) graph below illustrates the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/94a9bf07-1d45-47db-9fa5-b0786263ab26)

Figure_9. "ROC (Receiver Operating Characteristic) for the result of Decision Tree Classifier"

Departure Delay is the biggest contributor in the 2nd model, whereas it was the 4th largest contributor in the 1st model.
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/b7e77424-449e-45ec-84b3-7fd7fb9486a4)

Figure_10. "Permutation Feature Importance for the of Decision Tree Classifier"

**XGBClassifier vs Decision Tree Classifier**
![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/47d84808-5e7a-4899-86e3-591a2ea3e58d)

Figure_11. "Models Comparison"

