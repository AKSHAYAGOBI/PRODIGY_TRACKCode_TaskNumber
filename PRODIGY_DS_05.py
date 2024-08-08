import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset 
rta_data = pd.read_csv('/content/cleaned_rta_data.csv') # Load the cleaned dataset

# Summary statistics for key variables
print(rta_data['Accident_severity'].value_counts())
print(rta_data['Weather_conditions'].value_counts())
print(rta_data['Road_surface_conditions'].value_counts())
print(rta_data['Time_of_day'].value_counts())

# Group analysis: Accident severity by road and weather conditions
severity_by_weather = rta_data.groupby('Weather_conditions')['Accident_severity'].value_counts(normalize=True).unstack()
severity_by_road = rta_data.groupby('Road_surface_conditions')['Accident_severity'].value_counts(normalize=True).unstack()

# Visualization: Bar charts for accident distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=rta_data, x='Weather_conditions', hue='Accident_severity')
plt.title('Accident Severity by Weather Conditions')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(data=rta_data, x='Road_surface_conditions', hue='Accident_severity')
plt.title('Accident Severity by Road Surface Conditions')
plt.xticks(rotation=45)
plt.show()

# Time of day analysis
plt.figure(figsize=(10, 5))
sns.countplot(data=rta_data, x='Time_of_day', hue='Accident_severity')
plt.title('Accident Severity by Time of Day')
plt.show()
