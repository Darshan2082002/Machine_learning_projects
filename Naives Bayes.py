import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


df=pd.read_csv("spam(in).csv")
print(df.columns)
X = df['B']   
y = df['A']   


plt.figure(figsize=(6,4))
sns.countplot(x=y, palette="pastel")
plt.title("Distribution of Messages (Spam vs Ham)")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.show()


df['message_length'] = df['B'].apply(lambda x: len(str(x)))
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='message_length', hue='A', bins=30, kde=True, palette="muted")
plt.title("Message Length Distribution by Category")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)

print("\nModel Evaluation")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


example = ["Congratulations! You've won a free prize, click here!"]
example_tfidf = vectorizer.transform(example)
prediction = model.predict(example_tfidf)
print("\nExample prediction:", prediction[0])