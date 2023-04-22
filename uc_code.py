import pandas as pd
import numpy as np
from tableauscraper import TableauScraper as TS
from dash import Dash, html, Input, Output, dash_table
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import sklearn as sk
import sklearn.linear_model
import random

# Only 2021 for data
url = "https://visualizedata.ucop.edu/t/Public/views/AdmissionsDataTable/TREthbyYr?:embed_code_version=3&:embed=y&:loadOrderID=0&:display_spinner=no&:showAppBanner=false&:display_count=n&:showVizHome=n&:origin=viz_share_link"
ts = TS()
ts.loads(url)
workbook = ts.getWorkbook()

for t in workbook.worksheets:
    print(f"worksheet name : {t.name}") #show worksheet name
    print(t.data) #show dataframe for this worksheet
   
mainframe = t.data
mainframe.drop(mainframe.columns[[1, 2, 3, 4, 7, 8, 11]], axis=1, inplace=True)
mainframe.rename(columns = {'School-value': 'School', 'County-alias': 'County', 'City-value': 'City', 'Count-alias': 'Type', 'Uad Uc Ethn 6 Cat-value': 'Eth','SUM(Pivot Field Values)-alias':'Value'}, inplace = True)
mainframe.loc[mainframe["Value"] == "%null%", "Value"] = 0

# We may want to get rid of the 'Unknown' ethnicity
enr_frame = mainframe.loc[(mainframe['Type'] == 'Enr') & (mainframe['Eth'] != 'All')] # This is cuz everytime you sort for what eth has the most, 'All' will return every time.
adm_frame = mainframe.loc[(mainframe['Type'] == 'Adm') & (mainframe['Eth'] != 'All')]
app_frame = mainframe.loc[(mainframe['Type'] == 'App') & (mainframe['Eth'] != 'All')]
tot_frame = mainframe[mainframe['Eth'] == 'All']

# Resetting index so we can properly divide the columns later
enr_frame = enr_frame.reset_index()
adm_frame = adm_frame.reset_index()
app_frame = app_frame.reset_index()

# Getting rid of index col
enr_frame.drop(enr_frame.columns[[0]],axis=1,inplace=True) 
adm_frame.drop(adm_frame.columns[[0]], axis = 1, inplace = True)
app_frame.drop(app_frame.columns[[0]],axis=1,inplace=True)

# Switching to float for division later ,see line 50
app_frame['Value'] = pd.to_numeric(app_frame['Value'])
adm_frame['Value'] = pd.to_numeric(adm_frame['Value'])
enr_frame['Value'] = pd.to_numeric(adm_frame['Value'])

int_enr = mainframe.loc[(mainframe['Type'] == 'Enr') & (mainframe['Eth'] == 'International')] # For both enroll and international students
int_enr.sort_values(by=['Value'], ascending = False)
tot_frame[tot_frame.Type == 'App'].sort_values(by=['Value'], ascending = False) # Which colleges generally get the most students admitted

enr_frame.sort_values(by=['Value'], ascending = False) # I want to create a ratio for these so that we can see what schools have the best admit rates, or most of what ethnicity.

# Right now, we can produce a lot of descriptive statistics. We need to produce ratios of different ethnicities for each college.
# For example, using ALLAN HANCOCK COLLEGE value for 'All', I want to divide each individual ethnicity by this value. 

# This gives us a rate. The rate is the percentage of students that were admitted out of those who applied for each eth at each school.
# The idea is that we could see which ethnicity had the highest rate and use that for our stats. 

# Rate Column
pd.set_option('mode.use_inf_as_na', True)
app_frame['Value'] = pd.to_numeric(app_frame['Value'])
adm_frame['Value'] = pd.to_numeric(adm_frame['Value'])
adm_frame['Rate'] = adm_frame['Value']/app_frame['Value']
adm_frame = adm_frame.fillna(0)

# Getting rid of unknowns. Better to do it after we calculated rate to not mess with positioning
adm_frame = adm_frame[adm_frame['Eth']!= 'Unknown']
app_frame = app_frame[app_frame['Eth']!= 'Unknown']
enr_frame = enr_frame[enr_frame['Eth']!= 'Unknown']

# Adm Max Frame
adm _max = adm_frame.sort_values('Value', ascending=False).drop_duplicates('School').sort_index() # For each school, it returns the ethnicity group who got the most admits. 
adm_max[adm_max.School == 'DIABLO VALLEY COLLEGE'] # Just one instance for a school
adm_max.Eth.value_counts() # This is the count of what ethnicity each college's max amount of admits is equal to.

# Both below were just to see what it looked like 
adm_frame[adm_frame.Eth == 'American Indian'].sort_values(by = ['Rate'], ascending = False).head(20)
adm_frame.sort_values('Rate', ascending=False).head(10)

adm_frame.loc[adm_frame['Rate'] == 1.0] # Colleges with admit rate of 100% per ethnicity
adm_frame.loc[adm_frame['Rate'] == 1.0]['Eth'].value_counts() # Counts how many time each eth appears for those that had 100% admit rate. Good for desc stats.

# GPA
url = "https://visualizedata.ucop.edu/t/Public/views/AdmissionsDataTable/TRGPAbyYr?:embed_code_version=3&:embed=y&:loadOrderID=0&:display_spinner=no&:showAppBanner=false&:display_count=n&:showVizHome=n&:origin=viz_share_link"
ts = TS()
ts.loads(url)
workbook = ts.getWorkbook()

for t in workbook.worksheets:
    print(f"worksheet name : {t.name}") #show worksheet name
    print(t.data) #show dataframe for this worksheet
    
gpa = t.data
gpa.drop(gpa.columns[[1, 2, 4, 5, 6, 8]], axis=1, inplace=True)
gpa.rename(columns = {'School-value': 'School', 'City-alias': 'City', 'County-alias': 'County', 'Measure Names-alias': 'Type', 'Measure Values-alias':'GPA'}, inplace = True)

enr_gpa = gpa[gpa.Type == 'Enrl GPA']
adm_gpa = gpa[gpa.Type == 'Adm GPA']
app_gpa = gpa[gpa.Type == 'App GPA']
adm_gpa = adm_gpa[['School', 'County', 'City', 'Type', 'GPA']]
adm_gpa['Type'] = adm_gpa['Type'].str.replace('GPA', '')

df_all = adm_gpa.merge(adm_max.drop_duplicates(), on=['School', 'County', 'City'], 
                   how='left', indicator=True)

df_all.drop(df_all.columns[[3, 9]], axis=1, inplace=True)
df_all.rename(columns = {'Type_y': 'Type'}, inplace = True)

final_frame = df_all
final_frame.Eth.value_counts()

# CHANGE TO FINAL FRAME
adm_max = adm_frame.sort_values('Value', ascending=False).drop_duplicates('School').sort_index()
adm_max[adm_max.School == 'FOOTHILL COLLEGE']
# adm_max is the high for each school, where eth is the max amount of students admitted

# To check whether or not the international count were outliers. 
one_outs = adm_frame.loc[adm_frame['Rate'] == 1.0]
inters = one_outs[one_outs['Eth'] == 'International']['School'] # The 11 schools with 100% international student admittance
result1 = adm_max[adm_max['School'].isin(list(inters))]
result1 # From this result, we see that all the schools with a 100% international student admit rate, do not have international students as their ethnicity with max admittance.  


## Plots

# Interactive Boxplot
fig = px.box(adm_frame, x="Rate", y="Eth", color="Eth", points = "all",
                 labels={
                     "Rate": "Admittance Rate",
                     "Eth": "Ethnicity",
                 },
                title="Boxplot of Admittance Rate Per Each Ethnicity")
fig.show()

# Changing for outliers, and the fact that some schools have zero from int and native american
nhk = adm_frame[adm_frame['Rate'] > 0.2]
fig = px.box(nhk, x="Rate", y="Eth", color="Eth", points = "all",
                 labels={
                     "Rate": "Admittance Rate",
                     "Eth": "Ethnicity",
                 },
                title="Boxplot of Admittance Rate Per Each Ethnicity")
fig.show()

test_frame = adm_frame.loc[adm_frame['Rate']>0.4]
# scuffed but we working on it
fig = px.scatter(test_frame, 
                 x = 'Value',
                 y = 'Rate',
                 template = 'plotly_dark',
                 color = 'Eth',
                 trendline = 'ols',
                 title = 'Admit Rate Per Different Ethnicities')
fig.update_layout(showlegend=True)
fig.show()

adm_frame['Race Label'] = adm_frame.apply(lambda row: label_race(row), axis=1)
test_frame = adm_frame.loc[adm_frame['Rate']>0]

# Gives an OLS line for each Ethnicity. It's all right
fig = px.scatter(adm_frame, 
                 x = 'Value',
                 y = 'Rate',
                 template = 'plotly_dark',
                 color = 'Eth',
                 trendline = 'ols',
                 title = 'Admit Rate Per Different Ethnicities')
fig.update_layout(showlegend=True)
fig.show()

# Interactive Table for the admit max frame # OUTDATED
fig = go.Figure(data=[go.Table(
    header=dict(values=list(adm_max.columns),
                fill_color='lavenderblush',
                align='left'),
    cells=dict(values=[adm_max.School, adm_max.County, adm_max.City, adm_max.Type, adm_max.Eth, adm_max.Value, adm_max.Rate],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(
    title_text = "Each CC's Most Admitted Ethnicity",
    title_font_size=30,
    font_family="Times New Roman",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="black",
)
fig.update_traces(cells_font=dict(size = 10))
fig.show()

sns.set(rc={'figure.figsize':(8,10)})
sns.pointplot(data=adm_frame, x="Rate", y="Value", hue="Eth", title = "Point Plot of Admitted Ethnicities")

# Kernal Density Estimate Plot
sns.kdeplot(data=adm_frame, x="Rate", hue="Eth", multiple="stack")

sns.displot(data=adm_frame, x="Rate", hue="Eth", col="Eth")

sns.pairplot(data=adm_frame, hue="Eth")

fig = go.Figure(data=[go.Table(
    header=dict(values=list(final_frame.columns),
                fill_color='lavenderblush',
                align='left'),
    cells=dict(values=[final_frame.School, final_frame.County, final_frame.City, final_frame.GPA, final_frame.Type, final_frame.Eth, final_frame.Value, final_frame.Rate],
               fill_color='lavender',
               align='left'))
])
fig.update_layout(
    title_text = "Each CC's Most Admitted Ethnicity",
    title_font_size=30,
    font_family="Times New Roman",
    font_color="black",
    title_font_family="Times New Roman",
    title_font_color="black",
)
fig.update_traces(cells_font=dict(size = 10))
fig.show()

another = adm_frame[adm_frame['Value'] != 0]
# Logistic Regression Shit
# Change another to adm_frame when done. Result: did not help
from sklearn.model_selection import train_test_split

train, test = train_test_split(adm_frame, stratify = adm_frame['Eth'], train_size = .75)
train.head()

## Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train[['Rate']], train['Eth'])
test.copy()
test['predicted'] = lr.predict(test[['Rate']])
lr.coef_, lr.intercept_
coef = np.vstack((lr.coef_.T, lr.intercept_))
coef
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
scores = sigmoid(test.iloc[:, 5:7] @ coef[:, :7])
scores = scores.set_axis([c+"-score" for c in lr.classes_],axis = 1)
log_frame = pd.concat((test,scores), axis = 1)
log_frame.head(5)
# The probability that the predicted ethnicity was equal to the total
len(log_frame['Eth']==log_frame['Predicted'])
(sum(log_frame['Eth'] == log_frame['Predicted']))/len(log_frame)
log_frame[log_frame['Eth'] == log_frame['Predicted']].Eth.value_counts()


# 4 equation, differs from the one above
lr = sk.linear_model.LogisticRegression()
fitted = lr.fit(adm_frame.Rate.values.reshape(-1, 1), adm_frame.Eth)
print(fitted.coef_)
print(fitted.intercept_)

# This is using the rates as the continuous variable and the ethnicity as the vars. Each coeff represents each ethnicity. # Kinda outdated

# LOGREG TABLE UR GONNA USE THIS
import statsmodels.formula.api as smf 
formula = "Rate ~ C(Eth)"
log_reg = smf.logit(formula, data=adm_frame).fit() # Might wanna see output is when we get rid of rows that have 0 observations of american indian or international. 
print(log_reg.summary())

# THIS WAS BAD CUZ ETH NEEDS TO BE INDEPENDENT NOT DEPENDENT
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
feature_cols = ['Rate']#, 'Value'
X = adm_frame[feature_cols] # Features # This needs to be eth and rate is Y
y = adm_frame.Eth # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Still not amazing

# KNN 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.2, random_state=42)
  
knn = KNeighborsClassifier(n_neighbors=7)
  
knn.fit(X_train, y_train)
  
# Predict on dataset which model has not seen before
#print(knn.predict(X_test))
print(knn.score(X_test, y_test)) # They all suck

# Precision table
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# IDK what i was doin
chisqt = pd.crosstab(adm_frame.Eth, adm_frame.Type, margins=True)
print(chisqt)

from scipy.stats import chi2_contingency 
import numpy as np
chisqt = pd.crosstab(adm_frame.Eth, adm_frame.Type, margins=True)

value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3]) # So ethnicity is independent of eachother due to P = 1
#Try to check for the useful features. Why are some related why are some important. Compare the results between each prediction test. 

# OLS F results
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Rate ~ Eth', data=adm_frame).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# Better table they do the same shit lol
from statsmodels.formula.api import ols

model = ols('Rate ~ C(Eth)', data=adm_frame) # Encoding
fitted_model = model.fit()
fitted_model.summary() 
