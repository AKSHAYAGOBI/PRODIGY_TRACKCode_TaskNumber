import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset, ensuring proper handling of delimiters and quotes
data = pd.read_csv('/content/bank-additional-full.csv', delimiter=';', quotechar='"')  # Adjust delimiter and quotechar if needed

# Check if 'y' column exists, printing column names for debugging
if 'y' not in data.columns:
    print("Columns in the dataset:", data.columns)
    raise KeyError("The 'y' column is missing from the dataset.")

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'y':  # Make sure to skip the target variable 'y'
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable 'y'
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert 'yes' to 1 and 'no' to 0

# Ensure 'y' is correctly encoded
if data['y'].dtype != 'int64':
    raise ValueError("The 'y' column has not been correctly converted to numerical format.")

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

