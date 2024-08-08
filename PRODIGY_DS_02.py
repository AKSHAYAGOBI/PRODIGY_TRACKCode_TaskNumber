import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('/content/bank-additional-full.csv')

# Encode categorical variables, EXCLUDING the target variable 'y'
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'y':  # Make sure to skip the target variable
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable 'y' AFTER encoding other categorical features
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0) 

# Define features (X) and target (y)
X = data.drop('y', axis=1)  
y = data['y'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
