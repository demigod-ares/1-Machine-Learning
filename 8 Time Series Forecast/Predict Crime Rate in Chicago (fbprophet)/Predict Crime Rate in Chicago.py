'''PREDICTING CRIME RATE IN CHICAGO USING FACEBOOK PROPHET 

# # STEP #0: PROBLEM STATEMENT
- Image Source: https://commons.wikimedia.org/wiki/File:Chicago_skyline,_viewed_from_John_Hancock_Center.jpg
- The Chicago Crime dataset contains a summary of the reported crimes occurred in the City of Chicago from 2001 to 2017. 
- Dataset has been obtained from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system.
- Dataset contains the following columns: 
     - ID: Unique identifier for the record.
     - Case Number: The Chicago Police Department RD Number (Records Division Number), which is unique to the incident.
     - Date: Date when the incident occurred.
     - Block: address where the incident occurred
     - IUCR: The Illinois Unifrom Crime Reporting code.
     - Primary Type: The primary description of the IUCR code.
     - Description: The secondary description of the IUCR code, a subcategory of the primary description.
     - Location Description: Description of the location where the incident occurred.
     - Arrest: Indicates whether an arrest was made.
     - Domestic: Indicates whether the incident was domestic-related as defined by the Illinois Domestic Violence Act.
     - Beat: Indicates the beat where the incident occurred. A beat is the smallest police geographic area – each beat has a dedicated police beat car. 
     - District: Indicates the police district where the incident occurred. 
     - Ward: The ward (City Council district) where the incident occurred. 
     - Community Area: Indicates the community area where the incident occurred. Chicago has 77 community areas. 
     - FBI Code: Indicates the crime classification as outlined in the FBI's National Incident-Based Reporting System (NIBRS). 
     - X Coordinate: The x coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. 
     - Y Coordinate: The y coordinate of the location where the incident occurred in State Plane Illinois East NAD 1983 projection. 
     - Year: Year the incident occurred.
     - Updated On: Date and time the record was last updated.
     - Latitude: The latitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
     - Longitude: The longitude of the location where the incident occurred. This location is shifted from the actual location for partial redaction but falls on the same block.
     - Location: The location where the incident occurred in a format that allows for creation of maps and other geographic operations on this data portal. This location is shifted from the actual location for partial redaction but falls on the same block.
Datasource: https://www.kaggle.com/currie32/crimes-in-chicago

- You must install fbprophet package as follows: pip install fbprophet     
- If you encounter an error, try: conda install -c conda-forge fbprophet

- Prophet is open source software released by Facebook’s Core Data Science team.
- Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. 
- Prophet works best with time series that have strong seasonal effects and several seasons of historical data. 

- For more information, please check this out: 
https://research.fb.com/prophet-forecasting-at-scale/
https://facebook.github.io/prophet/docs/quick_start.html#python-api''' 

# # STEP #1: IMPORTING DATA
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import numpy as np
# dataframes creation for both training and testing datasets 
# chicago_df_0 = pd.read_csv('Chicago_Crimes_2001_to_2004.csv', error_bad_lines=False, dtype={'Y Coordinate': np.float64, 'Latitude': np.float64})
chicago_df_1 = pd.read_csv('Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)
chicago_df_2 = pd.read_csv('Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)
chicago_df_3 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)
chicago_df = pd.concat([chicago_df_1, chicago_df_2, chicago_df_3], ignore_index=False, axis=0)
# chicago_df_0 = False
chicago_df_1 = False
chicago_df_2 = False
chicago_df_3 = False
# # STEP #2: EXPLORING THE DATASET  
# Let's see how many null elements are contained in the data
plt.figure(figsize=(10,10))
sns.heatmap(chicago_df.isnull(), cbar = False, cmap = 'YlGnBu')
# ID Case Number Date Block IUCR Primary Type Description Location Description Arrest Domestic Beat District Ward Community Area FBI Code X Coordinate Y Coordinate Year Updated On Latitude Longitude Location
chicago_df.drop(['Unnamed: 0', 'Case Number', 'Case Number', 'IUCR', 'X Coordinate', 'Y Coordinate','Updated On','Year', 'FBI Code', 'Beat','Ward','Community Area', 'Location', 'District', 'Latitude' , 'Longitude'], inplace=True, axis=1)
# check chicago_df
# Assembling a datetime by rearranging the dataframe column "Date". 
chicago_df.Date = pd.to_datetime(chicago_df.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date 
chicago_df.index = pd.DatetimeIndex(chicago_df.Date)
# check chicago_df
chicago_df['Primary Type'].value_counts()
chicago_df['Primary Type'].value_counts().index
plt.figure(figsize = (15, 10))
sns.countplot(y= 'Primary Type', data = chicago_df, order = chicago_df['Primary Type'].value_counts().index)
plt.figure(figsize = (15, 25))
sns.countplot(y= 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().index)
''' # Taking only top 15
chicago_df['Primary Type'].value_counts().iloc[:15]
chicago_df['Primary Type'].value_counts().iloc[:15].index
plt.figure(figsize = (15, 10))
sns.countplot(y= 'Primary Type', data = chicago_df, order = chicago_df['Primary Type'].value_counts().iloc[:15].index)
plt.figure(figsize = (15, 10))
sns.countplot(y= 'Location Description', data = chicago_df, order = chicago_df['Location Description'].value_counts().iloc[:15].index)'''
chicago_df.resample('Y').size()
# Resample is a Convenience method for frequency conversion and resampling of time series.
plt.plot(chicago_df.resample('Y').size())
plt.title('Crimes Count Per Year')
plt.xlabel('Years')
plt.ylabel('Number of Crimes')
chicago_df.resample('M').size()
# Resample is a Convenience method for frequency conversion and resampling of time series.
plt.plot(chicago_df.resample('M').size())
plt.title('Crimes Count Per Month')
plt.xlabel('Months')
plt.ylabel('Number of Crimes')
chicago_df.resample('Q').size()
# Resample is a Convenience method for frequency conversion and resampling of time series.
plt.plot(chicago_df.resample('Q').size())
plt.title('Crimes Count Per Quarter')
plt.xlabel('Quarters')
plt.ylabel('Number of Crimes')

# # STEP #3: PREPARING THE DATA
chicago_prophet = chicago_df.resample('M').size().reset_index()
chicago_prophet.columns = ['Date', 'Crime Count']

# # STEP #4: MAKE PREDICTIONS
chicago_prophet_df_final = chicago_prophet.rename(columns={'Date':'ds', 'Crime Count':'y'})
m = Prophet()
m.fit(chicago_prophet_df_final)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
figure = m.plot(forecast, xlabel='Date', ylabel='Crime Rate')
figure3 = m.plot_components(forecast)

