#!/usr/bin/env python
# coding: utf-8

# ## Sultan Mammadov

# # Predicting Flight Delays

# This dataset is collected from the Bureau of Transportation Statistics, Govt. of the USA.

# In[98]:


#Link to the dataset
#https://www.kaggle.com/datasets/omerkrbck/5-desicion-tree-model


# # Import Data

# In[100]:


# Data is Imbalanced. SMOTE was used to handle class imbalance problem.
# pip install imbalanced-learn # Commented because it is already installed


# In[103]:


#Importing relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import zscore

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost import XGBClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, f1_score, confusion_matrix, balanced_accuracy_score, roc_auc_score, classification_report, RocCurveDisplay, ConfusionMatrixDisplay

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

from sklearn.inspection import permutation_importance

from graphviz import Digraph


# In[104]:


# Read Data
df = pd.read_csv("df_EDA.csv")

df.info()


# # Data Cleaning

# In[105]:


df.drop("Unnamed: 0", axis = 1, inplace = True)
data = df.copy()

data.dtypes


# In[106]:


# Check NULL
data.isnull().sum()


# In[108]:


# Check NA
data.isna().count()


# In[109]:


# Drop some columns
data.drop(['Quarter', 'Month', 'ArrDelay', 'Month_Str', 'DayOfWeek_Str'], axis = 1, inplace = True)

data.head(3)


# In[110]:


# Grouping data by Day of Month and calculating the mean Flight_Status for each day
flight_status_by_day = data.groupby('DayofMonth')['Flight_Status'].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(flight_status_by_day['DayofMonth'], flight_status_by_day['Flight_Status'], color='green')

plt.title('Proportion of Delayed Flights by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Proportion of Delayed Flights')
plt.xticks(flight_status_by_day['DayofMonth'])  # Ensure all days are shown
plt.grid(axis='y', linestyle='--')

plt.show()


# In[111]:


# Filter for flights with no delay
delayed_flights = data[data['Flight_Status'] == 1]

# Count the number of delayed flights per airline
delayed_flights_counts = delayed_flights['Airlines'].value_counts().reset_index()
delayed_flights_counts.columns = ['Airlines', 'No_Delay_Count']

# Sorting the counts for better visualization
delayed_flights_counts_sorted = delayed_flights_counts.sort_values(by='No_Delay_Count', ascending=True)

# Creating the bar chart
plt.figure(figsize=(10, 8))
plt.barh(delayed_flights_counts_sorted['Airlines'], delayed_flights_counts_sorted['No_Delay_Count'], color='lightblue')
plt.title('Number of Delayed Flights by Airline')
plt.xlabel('Number of Delayed Flights')
plt.ylabel('Airlines')
plt.show()


# In[112]:


# Grouping data by DayofMonth and calculating the mean Flight_Status for each day
flight_status_by_day = data.groupby('DayofMonth')['Flight_Status'].mean().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(flight_status_by_day['DayofMonth'], flight_status_by_day['Flight_Status'], color='skyblue')

plt.title('Proportion of Delayed Flights by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Proportion of Delayed Flights')
plt.xticks(flight_status_by_day['DayofMonth'])  # Ensure all days are shown
plt.grid(axis='y', linestyle='--')

plt.show()


# In[113]:


# Row Count Before Outlier Removal
data['Flight_Status'].count()


# In[114]:


# Remove Outlier

from scipy.stats import zscore

# Calculate z-scores for relevant columns
data[['DepDelay_WO']] = zscore(data[['DepDelay']])

# Identify rows with outliers
outliers = ((data['DepDelay_WO'] < -3) | (data['DepDelay_WO'] > 3))

# Drop rows with outliers
data = data.drop(data[outliers].index)

# Drop the "z_score" column
data = data.drop(['DepDelay_WO'], axis=1)


# In[115]:


# Row Count After Outlier Removal
data['Flight_Status'].count()


# In[116]:


# Label Encoding
label_model = LabelEncoder()

data['FlightDate'] = label_model.fit_transform(data['FlightDate'])
data['Airlines'] = label_model.fit_transform(data['Airlines'])
data['OriginCityName'] = label_model.fit_transform(data['OriginCityName'])
data['DestCityName'] = label_model.fit_transform(data['DestCityName'])

#Generate new column Speed
data['Speed'] = data['Distance'] / data['AirTime']

data.head(1)


# # Visualisation of Preprocessed Data

# In[118]:


# Visualise data with histogram
data.hist(alpha = 0.7, bins = 50, figsize = (14, 10))
plt.subplots_adjust(hspace = 0.6)
plt.show()


# In[119]:


# Correlation Matrix
correlation_matrix = data.iloc[:,2:18].corr()

plt.figure(figsize = (12, 8))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrix for Flight Delays')
plt.show()


# # Preparation Before Models

# In[121]:


#Assign x and y values

x = data.drop(['Flight_Status'], axis = 1)
y = data[['Flight_Status']]

x.head(2)


# In[122]:


# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[123]:


# SMOTE to balance the dataset

smote = SMOTE(random_state = 42)
smote_x_train, smote_y_train = smote.fit_resample(x_train, y_train)


# # 1st Model: XGBClassifier

# In[124]:


# Fit and Predict
xgb_model = XGBClassifier(max_depth = 10)
xgb_model.fit(smote_x_train, smote_y_train)
y_pred = xgb_model.predict(x_test)


# # Measue and Visualize Performance

# In[126]:


# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', round(accuracy, 2))

# Calculating the F1 Score
f1 = round(f1_score(y_test, y_pred),2)
print("F1 Score is " + str(f1))

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:', conf_matrix)


# In[127]:


# Visualize Confusion Matrix
plt.figure(figsize = (7, 5))
ax = sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'viridis', cbar = False, annot_kws={'size':16})
plt.title('Confusion Matrix', fontsize = 20)
plt.ylabel('True Label', fontsize = 16)
plt.xlabel('Predicted Label', fontsize = 16)
ax.set_xticklabels(['Not Delayed (0)', 'Delayed (1)'], fontsize=14)
ax.set_yticklabels(['Not Delayed  (0)', 'Delayed (1)'], fontsize=14, rotation=0)
plt.show()


# In[128]:


#Visualise Tree
fig, ax = plt.subplots(figsize = (30, 30))
xgb.plot_tree(xgb_model, num_trees = 0, ax = ax, class_names = data['Flight_Status'].unique())


# In[131]:


# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the Area Under Curve
auc_score = roc_auc_score(y_test, y_pred)
print('Area Under Curve:', round(auc_score, 2))


# In[132]:


# Plotting the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=20)
plt.legend(loc="lower right", fontsize=16)
plt.gca().set_facecolor('lightgray')
plt.grid(True)
plt.show()


# In[133]:


# TPR FPR
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, y_pred).ravel()

# Calculating True Positive Rate (TPR) and False Positive Rate (FPR)
tpr = round(true_positive / (true_positive + false_negative),2)  # TPR = TP / (TP + FN)
fpr = round(false_positive / (false_positive + true_negative),2) # FPR = FP / (FP + TN)

# Creating a table for display
results = {
    "Metrics": ["True Positive Rate (TPR)", "False Positive Rate (FPR)"],
    "Values": [tpr, fpr]
}

results_df = pd.DataFrame(results)

# Plotting the table
fig, ax = plt.subplots(figsize=(5, 2))  # set size frame
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc = 'center', loc='center',
         colColours=["palegreen", "paleturquoise"])

plt.title("Calculated TPR and FPR", fontsize=16, color="darkblue")
plt.gca().set_facecolor('lightgray')
plt.show()


# In[134]:


# Feature Importance

# Calculate permutation feature importance
perm_importance = permutation_importance(xgb_model, x_test, y_test, n_repeats=30, random_state=42)

# Organize results into a DataFrame for easier visualization
perm_importance_df = pd.DataFrame({'feature_names': x.columns, 'importance_mean': perm_importance.importances_mean})

# Sort by importance
perm_importance_df = perm_importance_df.sort_values(by='importance_mean', ascending=False)
perm_importance_df


# In[135]:


# Assuming perm_importance is computed earlier
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(13, 7))

# Use sorted indices to sort feature names and their importance values
plt.barh(x_test.columns[sorted_idx], perm_importance.importances_mean[sorted_idx], color='skyblue')

plt.xlabel("Permutation Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.title("Permutation Feature Importance", fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()  # Adjust the layout to make room for the labels

# Update the value labels to reflect the sorted bars
for index, value in enumerate(perm_importance.importances_mean[sorted_idx]):
    plt.text(value, index, f"{value:.4f}", va='center')

plt.show()


# In[136]:


# Input Output Graph
dot = Digraph(comment='Flight Status Influencers', format='png')
dot.attr(rankdir='LR', size='10')

# Setting the background color
dot.attr(bgcolor='gray')

# Customizing node attributes
dot.attr('node', shape='box', style='filled', color='lightgrey', fontname='Helvetica', fillcolor='white')

# Inputs
dot.node('K', 'Day of Month', fillcolor='skyblue')
dot.node('E', 'Day Of Week', fillcolor='skyblue')
dot.node('F', 'Flight Date', fillcolor='skyblue')
dot.node('B', 'Airlines', fillcolor='skyblue')
dot.node('G', 'Origin City Name', fillcolor='skyblue')
dot.node('H', 'Destination City Name', fillcolor='skyblue')
dot.node('A', 'Departure Delay', fillcolor='skyblue')
dot.node('T', 'Air Time', fillcolor='skyblue')
dot.node('C', 'Distance', fillcolor='skyblue')
dot.node('X', 'Speed', fillcolor='brown')

# Output
dot.attr('node', shape='ellipse', fillcolor='tomato')
dot.node('D', 'Flight Status')

# Customizing edge attributes
dot.attr('edge', color='blue', arrowhead='vee', fontname='Helvetica')

# Adding edges with arrows
dot.edge('K', 'D')
dot.edge('E', 'D')
dot.edge('F', 'D')
dot.edge('B', 'D')
dot.edge('G', 'D')
dot.edge('H', 'D')
dot.edge('A', 'D')
dot.edge('T', 'D')
dot.edge('C', 'D')
dot.edge('X', 'D')

# Render the graph to a file and automatically open it
dot.render('flight_status_influencers_background', view=True)


# # Data Preprocessing Before Model (Decision Tree)

# In[137]:


unique_cities = df['OriginCityName'].unique()

city_to_state = {
    'Atlanta, GA': 'Georgia','Fort Lauderdale, FL': 'Florida','Jackson/Vicksburg, MS': 'Mississippi','Richmond, VA': 'Virginia','Minneapolis, MN': 'Minnesota','Raleigh/Durham, NC': 'North Carolina','Nashville, TN': 'Tennessee','Indianapolis, IN': 'Indiana','New York, NY': 'New York','Savannah, GA': 'Georgia','Fayetteville, AR': 'Arkansas','San Antonio, TX': 'Texas','Tampa, FL': 'Florida','Salt Lake City, UT': 'Utah','Hartford, CT': 'Connecticut','Jacksonville, FL': 'Florida','Boston, MA': 'Massachusetts','Fort Myers, FL': 'Florida','Seattle, WA': 'Washington','Harrisburg, PA': 'Pennsylvania','Miami, FL': 'Florida','Pittsburgh, PA': 'Pennsylvania','Orlando, FL': 'Florida','Charlotte, NC': 'North Carolina','Columbia, SC': 'South Carolina','Newark, NJ': 'New Jersey','Buffalo, NY': 'New York','Philadelphia, PA': 'Pennsylvania','Cincinnati, OH': 'Ohio','Las Vegas, NV': 'Nevada','Portland, OR': 'Oregon','Washington, DC': 'District of Columbia','Detroit, MI': 'Michigan','Norfolk, VA': 'Virginia','Sioux Falls, SD': 'South Dakota','Los Angeles, CA': 'California','Dallas/Fort Worth, TX': 'Texas','Memphis, TN': 'Tennessee','San Juan, PR': 'Puerto Rico',
    'Rochester, NY': 'New York','Chicago, IL': 'Illinois','Louisville, KY': 'Kentucky','Columbus, OH': 'Ohio','Pensacola, FL': 'Florida','Phoenix, AZ': 'Arizona','Charleston, SC': 'South Carolina','Dayton, OH': 'Ohio','Portland, ME': 'Maine','Jackson, WY': 'Wyoming','Santa Ana, CA': 'California','Kansas City, MO': 'Missouri','Long Beach, CA': 'California','Boise, ID': 'Idaho','New Orleans, LA': 'Louisiana','Tulsa, OK': 'Oklahoma','Charlotte Amalie, VI': 'Virgin Islands','Grand Rapids, MI': 'Michigan','Madison, WI': 'Wisconsin','Austin, TX': 'Texas','Christiansted, VI': 'Virgin Islands','Oakland, CA': 'California','Denver, CO': 'Colorado','Baltimore, MD': 'Maryland','Little Rock, AR': 'Arkansas','Houston, TX': 'Texas','Panama City, FL': 'Florida','Daytona Beach, FL': 'Florida','Sarasota/Bradenton, FL': 'Florida','San Diego, CA': 'California','Valparaiso, FL': 'Florida','Huntsville, AL': 'Alabama','San Jose, CA': 'California','West Palm Beach/Palm Beach, FL': 'Florida','St. Louis, MO': 'Missouri','Myrtle Beach, SC': 'South Carolina','Cleveland, OH': 'Ohio','Ontario, CA': 'California','Omaha, NE': 'Nebraska','San Francisco, CA': 'California','Milwaukee, WI': 'Wisconsin','Anchorage, AK': 'Alaska','Fairbanks, AK': 'Alaska','White Plains, NY': 'New York','Birmingham, AL': 'Alabama','Knoxville, TN': 'Tennessee',
    'Oklahoma City, OK': 'Oklahoma','Hayden, CO': 'Colorado','Chattanooga, TN': 'Tennessee','Bozeman, MT': 'Montana','Des Moines, IA': 'Iowa','Key West, FL': 'Florida','Wichita, KS': 'Kansas','Palm Springs, CA': 'California','Roanoke, VA': 'Virginia','Greensboro/High Point, NC': 'North Carolina','Tallahassee, FL': 'Florida','Spokane, WA': 'Washington','Appleton, WI': 'Wisconsin','Gainesville, FL': 'Florida','Kalispell, MT': 'Montana','Melbourne, FL': 'Florida','Greer, SC': 'South Carolina','Lexington, KY': 'Kentucky','Baton Rouge, LA': 'Louisiana','Asheville, NC': 'North Carolina','Reno, NV': 'Nevada','Gulfport/Biloxi, MS': 'Mississippi','Springfield, MO': 'Missouri','Dallas, TX': 'Texas','Providence, RI': 'Rhode Island','Sacramento, CA': 'California','Albany, NY': 'New York','Missoula, MT': 'Montana','Syracuse, NY': 'New York','Eagle, CO': 'Colorado','Honolulu, HI': 'Hawaii','Lihue, HI': 'Hawaii','Kahului, HI': 'Hawaii','Kona, HI': 'Hawaii','Albuquerque, NM': 'New Mexico','Tucson, AZ': 'Arizona','El Paso, TX': 'Texas','Burlington, VT': 'Vermont','Fargo, ND': 'North Dakota','Green Bay, WI': 'Wisconsin','Montrose/Delta, CO': 'Colorado','Fayetteville, NC': 'North Carolina','Billings, MT': 'Montana','State College, PA': 'Pennsylvania','Harlingen/San Benito, TX': 'Texas','Trenton, NJ': 'New Jersey',
    'Colorado Springs, CO': 'Colorado','Islip, NY': 'New York','Newburgh/Poughkeepsie, NY': 'New York','Burbank, CA': 'California','Grand Junction, CO': 'Colorado','Cedar Rapids/Iowa City, IA': 'Iowa','Bloomington/Normal, IL': 'Illinois','Fresno, CA': 'California','Durango, CO': 'Colorado','Wilmington, DE': 'Delaware','Sanford, FL': 'Florida','Flint, MI': 'Michigan','Concord, NC': 'North Carolina','Punta Gorda, FL': 'Florida','St. Petersburg, FL': 'Florida','Evansville, IN': 'Indiana','Niagara Falls, NY': 'New York','Peoria, IL': 'Illinois','St. Cloud, MN': 'Minnesota','Bangor, ME': 'Maine','Portsmouth, NH': 'New Hampshire','Fort Wayne, IN': 'Indiana','Medford, OR': 'Oregon','Stockton, CA': 'California','Moline, IL': 'Illinois','Santa Maria, CA': 'California','Toledo, OH': 'Ohio','Plattsburgh, NY': 'New York','Allentown/Bethlehem/Easton, PA': 'Pennsylvania','Shreveport, LA': 'Louisiana','Grand Island, NE': 'Nebraska','Rapid City, SD': 'South Dakota','Belleville, IL': 'Illinois','Rockford, IL': 'Illinois','Idaho Falls, ID': 'Idaho','Bismarck/Mandan, ND': 'North Dakota','Hagerstown, MD': 'Maryland','Grand Forks, ND': 'North Dakota','Bristol/Johnson City/Kingsport, TN': 'Tennessee','Elmira/Corning, NY': 'New York','Provo, UT': 'Utah','Minot, ND': 'North Dakota','Springfield, IL': 'Illinois','South Bend, IN': 'Indiana','Mission/McAllen/Edinburg, TX': 'Texas','Pasco/Kennewick/Richland, WA': 'Washington','Bellingham, WA': 'Washington','Traverse City, MI': 'Michigan','Great Falls, MT': 'Montana','Monterey, CA': 'California','Ogden, UT': 'Utah','Amarillo, TX': 'Texas','Laredo, TX': 'Texas','Eugene, OR': 'Oregon','Owensboro, KY': 'Kentucky','Ashland, WV': 'West Virginia','Clarksburg/Fairmont, WV': 'West Virginia','Bend/Redmond, OR': 'Oregon',
    'Hilo, HI': 'Hawaii','Pago Pago, TT': 'American Samoa','Garden City, KS': 'Kansas','Midland/Odessa, TX': 'Texas','Brownsville, TX': 'Texas','Lake Charles, LA': 'Louisiana','Longview, TX': 'Texas','Lafayette, LA': 'Louisiana','Wichita Falls, TX': 'Texas','Champaign/Urbana, IL': 'Illinois','Kalamazoo, MI': 'Michigan','Waterloo, IA': 'Iowa','Fort Smith, AR': 'Arkansas','La Crosse, WI': 'Wisconsin','Lawton/Fort Sill, OK': 'Oklahoma','Abilene, TX': 'Texas','Tyler, TX': 'Texas','Killeen, TX': 'Texas','Dubuque, IA': 'Iowa','College Station/Bryan, TX': 'Texas','San Angelo, TX': 'Texas','Manhattan/Ft. Riley, KS': 'Kansas','Marquette, MI': 'Michigan','Monroe, LA': 'Louisiana','Stillwater, OK': 'Oklahoma','Augusta, GA': 'Georgia','Lubbock, TX': 'Texas','Texarkana, AR': 'Arkansas','Corpus Christi, TX': 'Texas','Mosinee, WI': 'Wisconsin','Waco, TX': 'Texas','Scranton/Wilkes-Barre, PA': 'Pennsylvania','Alexandria, LA': 'Louisiana','Montgomery, AL': 'Alabama','Mobile, AL': 'Alabama','Rochester, MN': 'Minnesota','Columbia, MO': 'Missouri','San Luis Obispo, CA': 'California','Lansing, MI': 'Michigan','Beaumont/Port Arthur, TX': 'Texas','Wilmington, NC': 'North Carolina','Del Rio, TX': 'Texas','Roswell, NM': 'New Mexico','Atlantic City, NJ': 'New Jersey','Aguadilla, PR': 'Puerto Rico','Latrobe, PA': 'Pennsylvania','Charleston/Dunbar, WV': 'West Virginia','Manchester, NH': 'New Hampshire','Akron, OH': 'Ohio','Newport News/Williamsburg, VA': 'Virginia','Lynchburg, VA': 'Virginia','Erie, PA': 'Pennsylvania','New Bern/Morehead/Beaufort, NC': 'North Carolina',
    'Ithaca/Cortland, NY': 'New York','Charlottesville, VA': 'Virginia','Columbus, GA': 'Georgia','Jacksonville/Camp Lejeune, NC': 'North Carolina','Yakima, WA': 'Washington','Redding, CA': 'California','Santa Rosa, CA': 'California','Walla Walla, WA': 'Washington','Santa Barbara, CA': 'California','Everett, WA': 'Washington','Pullman, WA': 'Washington','Wenatchee, WA': 'Washington','Sun Valley/Hailey/Ketchum, ID': 'Idaho','Helena, MT': 'Montana','Dillingham, AK': 'Alaska','King Salmon, AK': 'Alaska','Duluth, MN': 'Minnesota','Hilton Head, SC': 'South Carolina','Worcester, MA': 'Massachusetts','Brunswick, GA': 'Georgia','Albany, GA': 'Georgia','Dothan, AL': 'Alabama','Saginaw/Bay City/Midland, MI': 'Michigan','Valdosta, GA': 'Georgia','Columbus, MS': 'Mississippi','Binghamton, NY': 'New York','Gunnison, CO': 'Colorado','Greenville, NC': 'North Carolina','Watertown, NY': 'New York','Salisbury, MD': 'Maryland','Florence, SC': 'South Carolina','Bethel, AK': 'Alaska','Kodiak, AK': 'Alaska','Barrow, AK': 'Alaska','Juneau, AK': 'Alaska','Ketchikan, AK': 'Alaska','Sitka, AK': 'Alaska','Kotzebue, AK': 'Alaska','Cordova, AK': 'Alaska','Wrangell, AK': 'Alaska','Petersburg, AK': 'Alaska','Nome, AK': 'Alaska','Deadhorse, AK': 'Alaska','Yakutat, AK': 'Alaska','Adak Island, AK': 'Alaska','Ponce, PR': 'Puerto Rico','Aspen, CO': 'Colorado','Yuma, AZ': 'Arizona','Flagstaff, AZ': 'Arizona','Santa Fe, NM': 'New Mexico','Bakersfield, CA': 'California','Arcata/Eureka, CA': 'California','St. George, UT': 'Utah','Lewiston, ID': 'Idaho','Escanaba, MI': 'Michigan','International Falls, MN': 'Minnesota','Aberdeen, SD': 'South Dakota','Sault Ste. Marie, MI': 'Michigan','Twin Falls, ID': 'Idaho','Hibbing, MN': 'Minnesota','Williston, ND': 'North Dakota','Rhinelander, WI': 'Wisconsin',
    'Iron Mountain/Kingsfd, MI': 'Michigan','Butte, MT': 'Montana','Pellston, MI': 'Michigan','Bemidji, MN': 'Minnesota','Brainerd, MN': 'Minnesota','Cedar City, UT': 'Utah','Casper, WY': 'Wyoming','Elko, NV': 'Nevada','Pocatello, ID': 'Idaho','Alpena, MI': 'Michigan','Lincoln, NE': 'Nebraska','Ogdensburg, NY': 'New York','Jamestown, ND': 'North Dakota','Staunton, VA': 'Virginia','Pueblo, CO': 'Colorado','Hattiesburg/Laurel, MS': 'Mississippi','Sheridan, WY': 'Wyoming','Prescott, AZ': 'Arizona','Cape Girardeau, MO': 'Missouri','Hays, KS': 'Kansas','Pierre, SD': 'South Dakota','Rock Springs, WY': 'Wyoming','Moab, UT': 'Utah','Watertown, SD': 'South Dakota','Cheyenne, WY': 'Wyoming','Liberal, KS': 'Kansas','Victoria, TX': 'Texas','Meridian, MS': 'Mississippi','Johnstown, PA': 'Pennsylvania','Kearney, NE': 'Nebraska','Bishop, CA': 'California','Hobbs, NM': 'New Mexico','Cody, WY': 'Wyoming','North Bend/Coos Bay, OR': 'Oregon','Dodge City, KS': 'Kansas','Lewisburg, WV': 'West Virginia','Scottsbluff, NE': 'Nebraska','Fort Leonard Wood, MO': 'Missouri','North Platte, NE': 'Nebraska','Hancock/Houghton, MI': 'Michigan','Devils Lake, ND': 'North Dakota','Joplin, MO': 'Missouri','Eau Claire, WI': 'Wisconsin','Muskegon, MI': 'Michigan','Decatur, IL': 'Illinois','Paducah, KY': 'Kentucky','Sioux City, IA': 'Iowa','Laramie, WY': 'Wyoming','Salina, KS': 'Kansas','Mason City, IA': 'Iowa','Fort Dodge, IA': 'Iowa','Vernal, UT': 'Utah','Riverton/Lander, WY': 'Wyoming','Gillette, WY': 'Wyoming','Alamosa, CO': 'Colorado','Guam, TT': 'Guam','Saipan, TT': 'Northern Mariana Islands','Dickinson, ND': 'North Dakota','Presque Isle/Houlton, ME': 'Maine','West Yellowstone, MT': 'Montana','Nantucket, MA': 'Massachusetts',
    'Gustavus, AK': 'Alaska','Hyannis, MA': 'Massachusetts',"Martha's Vineyard, MA": 'Massachusetts','Branson, MO': 'Missouri'
}


# In[138]:


df['Originstate'] = df['OriginCityName'].map(city_to_state)
df['Deststate'] = df['DestCityName'].map(city_to_state)


# In[142]:


df['Deststate'].nunique()


# In[143]:


df['Originstate'].nunique()


# In[144]:


df1 = df.copy()
df1 = df1.drop(['Quarter', 'FlightDate', 'Month_Str', 'DayOfWeek_Str', 'OriginCityName', 'DestCityName', 'ArrDelay' ], axis=1)


# In[145]:


df1.head(1)


# In[146]:


df1.Airlines.unique()


# In[147]:


# check for missing values
df1.isnull().sum()


# In[148]:


df1.dropna()


# In[149]:


df1.isnull().sum()


# In[150]:


# Making dummies

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform Airlines column
df1['Airlines'] = label_encoder.fit_transform(df1['Airlines'])

# Fit and transform Originstate column
df1['Originstate'] = label_encoder.fit_transform(df1['Originstate'])

# Fit and transform Deststate column
df1['Deststate'] = label_encoder.fit_transform(df1['Deststate'])


# In[151]:


df1.shape


# In[152]:


df1.describe().T


# In[153]:


# Count the occurrences of each category in "Flight_Status"
flight_status_counts = df1["Flight_Status"].value_counts()

# Plot the distribution of Flight_Status
plt.figure(figsize=(8, 6))
flight_status_counts.plot(kind='bar', color='skyblue')
plt.title('Flight Status Distribution')
plt.xlabel('Flight Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[154]:


# Splitting the independant (x) and dependant variables (y)
x = df1.drop(columns ="Flight_Status")
y = df1["Flight_Status"]


# In[155]:


# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 42)


# In[156]:


# SMOTE

# Apply SMOTE to the training data only
smote = SMOTE(random_state = 42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)


# In[ ]:


# Get unique values and their counts
values, counts = np.unique(y_train_smote, return_counts=True)

# Create bar chart
plt.bar([str(x) for x in values], counts)
plt.xlabel("target attribute values")
plt.ylabel("Frequency")
plt.title("Training data after SMOTE") 


# # 2nd Model: Decision Tree Classifier

# In[157]:


# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state = 42)
clf.fit(x_train_smote, y_train_smote)


# # Measue and Visualize Performance

# In[158]:


print("Classification Report:")

# Make predictions
y_pred_smote = clf.predict(x_test)

print(classification_report(y_test, y_pred_smote))
print("Accuracy:", accuracy_score(y_test, y_pred_smote))


# In[159]:


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_smote)

# Display confusion matrix
print("Confusion Matrix:")
print(cm)


# In[160]:


# Display of the Confusion Matrix
tn, fp, fn, tp = cm.ravel() # returns flattened array
print("tn:" + str(tn), "fp:" + str(fp),"fn:" + str(fn), "tp:" + str(tp))

# Confusion matrix
cm_display = ConfusionMatrixDisplay(cm).plot()


# In[161]:


# Precision Score

print('Precision Score:', round(precision_score(y_test, y_pred_smote),3))


# In[162]:


# ROC Curve

# Predict Proba for SMOTE
y_pred_smote_prob = clf.predict_proba(x_test.values)[:,1]

fpr, tpr, _thresholds  = roc_curve(y_test, y_pred_smote_prob, pos_label = clf.classes_[1])

#by default, the positive class is clf.classes_[1]
roc_display = RocCurveDisplay(fpr = fpr, tpr = tpr, estimator_name = "decision tree").plot()
plt.title("ROC curve") 


# In[163]:


# Getting Feature Importances

feature_names = list(x_train.columns)

importances = clf.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
feature_names = [feature_names[i] for i in indices]

# Plot the feature importances as a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(x.shape[1]), importances[indices])  # Use barh for horizontal bar chart
plt.yticks(range(x.shape[1]), feature_names)  # Set yticks with feature names
plt.xlabel('Importance')
plt.gca().invert_yaxis()  # Invert y-axis to display most important features at the top
plt.show()


# In[166]:


# ROC Auc Score
print('ROC Auc Score:', round(roc_auc_score(y_test, y_pred_smote_prob), 3))


# In[172]:


# Limiting the depth of the tree

clf = DecisionTreeClassifier(max_depth = 3, random_state = 42)
clf.fit(x_train_smote, y_train_smote)


# In[ ]:


# Limitting the number of features displayed
plt.figure(figsize = (10, 6))
plot_tree(clf, feature_names = feature_names, filled = True, rounded = True, max_depth = 3)
plt.show()


# In[173]:


#Using Grid Search to find the best model

# Split your data into a smaller subset for grid search
x_train_subset, _, y_train_subset, _ = train_test_split(x_train_smote, y_train_smote, train_size=0.1, random_state=42)

# Define a smaller parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5,7,None],
    'min_samples_split': [2, 5,10],
    'min_samples_leaf': [1, 2,4]
}

# Create a decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Create GridSearchCV object
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', verbose=1)

# Fit the grid search to the smaller subset of data
grid_search.fit(x_train_subset, y_train_subset)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)


# In[174]:


# Create a Decision Tree classifier with the best parameters
best_dt_classifier = DecisionTreeClassifier(criterion='gini', 
                                            max_depth=None, 
                                            min_samples_leaf=4, 
                                            min_samples_split=10,
                                            random_state=42)

# Fit the classifier to the full training data
best_dt_classifier.fit(x_train_smote, y_train_smote)


# In[177]:


print("Classification Report:")

# Make predictions from the best model
y_pred_best = best_dt_classifier.predict(x_test)

print(classification_report(y_test, y_pred_best))

print("Accuracy:", round(accuracy_score(y_test, y_pred_best), 3))

