**Introduction:**
  
Flight delays are a significant concern for air transportation worldwide due to the financial losses they cause. 
In 2022, nearly a quarter of US flights were delayed, causing significant financial impact. The aviation industry faced costs estimated between $30 billion and $34 billion due to these delays and cancellations. 
These delays inconvenience both airlines and passengers, leading to increased expenses for food and lodging, stress, and distrust towards airlines. 
Airlines also face extra costs such as crew expenses, aircraft repositioning, and increased fuel consumption, which damage their reputation and reduce demand.

Delays occur for various reasons, including air congestion, weather conditions, mechanical issues, boarding difficulties, and airlines' inability to manage demand.

** Business Question:**
- Can passengers avoid delayed flights or predict delays before boarding?

Using Machine Learning (ML) algorithms, it is possible to predict flight delays to some extent. Different algorithms have varying accuracy and depend on the data provided.

In this project, I explored the XGBOOST CLASSIFIER model to predict flight delays. I also examined the most contributing features in the model.

**Objective:**
The goal of this project is straightforward: developing a model that predicts flight delays before they appear on the departure boards.

**Data Collection:**
The dataset, sourced from Kaggle, includes multi-year data spanning from 2009 to 2023.


** Model: XGBClassifier: **

The bar chart displays the proportion of delayed flights by day of the month, providing insight into the busiest days.

<img width="1190" height="590" alt="download" src="https://github.com/user-attachments/assets/8672b91e-0f31-46d6-830e-575d277eaa9d" />

Figure_1. "Proportion of Delayed Flights by Day of Month"

The bar chart displays the number of delayed flights by airline, highlighting the airlines with the most and least delays.

![image](https://github.com/SultanMammadov/Flight-Delays-Prediction/assets/126120167/45daadf7-30ec-4725-8c4c-d567689cde74)

Figure_2. "The number of Delayed Flights by airline"

The Big Four — American, Southwest, United, and Delta — dominate delays with 220K–335K each, while every other airline combined barely comes close. 
American Airlines sits at the top with roughly 335K delays, nearly 4x the next tier. 
Smaller carriers like Hawaiian and Allegiant barely register, mostly because they simply fly far fewer routes.


The graphs below display the distributions and trends of all variables in the dataset.

<img width="1167" height="913" alt="download" src="https://github.com/user-attachments/assets/700e6631-1722-4b76-b320-39f69c619534" />

Figure_3. "Distributions and trends of all variables"

Most flights are on time, with delays being the exception rather than the rule. The majority of flights are short trips — think under 3 hours and under 1,000 miles. 
And airlines stay pretty busy year-round with no single day standing out as dramatically busier than others.


The correlation matrix shows the relationship between different variables in a dataset. 
It displays correlation coefficients, which quantify the strength and direction of the linear relationship between pairs of variables. 
Longer flights obviously cover more miles — air time and distance move together almost perfectly at 0.99. 
If a flight leaves late, it almost always arrives late too, with departure delay and flight status strongly linked at 0.71. 
Speed also ties closely to distance (0.71) and air time (0.62), meaning faster planes tend to fly longer routes. 
Beyond that, things like the date (0.013), the airline (0.049), or which city you're flying from (-0.017) barely move the needle on whether a flight gets delayed.

<img width="1095" height="790" alt="download" src="https://github.com/user-attachments/assets/d4f9e1cf-a7e4-4ce9-b6cf-ae6163c20928" />

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

**Summary & Recommendations:**

From the Exploratory Data Analysis (EDA), it appears that Hawaiian Airlines and Allegiant Airlines are among the most reliable in terms of on-time arrivals.

However, it is crucial to note that these conclusions are based on data from the first quarter of 2022. This period might have been particularly favorable for these airlines and less so for others. Therefore, I recommend conducting a more comprehensive historical analysis by including data from the remaining nine months. This would provide a more balanced view, although it may require more computational resources to run the same models.

To address data imbalance, the SMOTE (Synthetic Minority Over-sampling Technique) method was utilized to generate synthetic data. Additionally, the GridSearchCV method was employed to select the optimal parameters for the models.

Based on the above comparison below, the XGBClassifier is recommended as the primary model for this dataset.
