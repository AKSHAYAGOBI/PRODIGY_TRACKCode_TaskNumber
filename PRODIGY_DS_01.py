import pandas as pd
import matplotlib.pyplot as plt

# Load the diabetes data file
diabetes_data = pd.read_csv('C:/diabetes.csv')

# Display the first few rows of the data
print(diabetes_data.head())

# Assuming you want to create a histogram of the 'Age' column
plt.figure(figsize=(10, 6))
plt.hist(diabetes_data['Age'].dropna(), bins=20, edgecolor='k')
plt.title('Age Distribution of Diabetes Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
