import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = pd.read_csv('C:/titanic3.csv')

# Fill missing values in 'age' with median
df.loc[:, 'age'] = df['age'].fillna(df['age'].median())

# Fill missing values in 'embarked' with mode
df.loc[:, 'embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Drop 'cabin' and 'boat' columns as they have a large number of missing values
df.drop(columns=['cabin', 'boat'], inplace=True)

# Drop rows with any remaining missing values
df.dropna(inplace=True)

# Display the first few rows of the cleaned dataset
print(df.head())

# Distribution of 'age'
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Survival rate by sex
sns.barplot(x='age',y='survived',data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by class
sns.barplot(x='pclass', y='survived', data=df)
plt.title('Survival Rate by Class')
plt.show()

# Age distribution by survival
sns.histplot(df[df['survived'] == 1]['age'], color='green', label='Survived', kde=True)
sns.histplot(df[df['survived'] == 0]['age'], color='red', label='Not Survived', kde=True)
plt.legend()
plt.title('Age Distribution by Survival')
plt.show()
