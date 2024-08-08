import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
training_data = pd.read_csv('/content/twitter_training.csv', header=None) 
validation_data = pd.read_csv('/content/twitter_validation.csv', header=None) 

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Handle missing values and ensure all entries are strings
training_data[3] = training_data[3].fillna('').astype(str) # Access column by index 3
validation_data[3] = validation_data[3].fillna('').astype(str) # Access column by index 3

# Apply preprocessing to the text columns in both datasets
training_data[3] = training_data[3].apply(preprocess_text)
validation_data[3] = validation_data[3].apply(preprocess_text)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a bar plot for sentiment distribution by topic in the training dataset
plt.figure(figsize=(12, 8))

# Access columns by index for grouping
sentiment_counts = training_data.groupby([1, 2]).size().unstack().fillna(0) 
sentiment_counts.plot(kind='bar', stacked=True, colormap='viridis', ax=plt.gca())

plt.title('Sentiment Distribution by Topic in Training Dataset')
plt.xlabel('Topic')
plt.ylabel('Number of Tweets')
plt.legend(title='Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()
