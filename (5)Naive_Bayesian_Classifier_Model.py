from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Categories
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# Load dataset (downloads automatically if internet is available)
train_data = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42,
    download_if_missing=True
)

test_data = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42,
    download_if_missing=True
)

# Convert text to TF-IDF directly (simpler method)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)

# Train model
model = MultinomialNB()
model.fit(X_train, train_data.target)

# Predict
predicted = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(test_data.target, predicted))
print("\nClassification Report:\n")
print(classification_report(test_data.target, predicted,
                            target_names=test_data.target_names))

print("\nConfusion Matrix:\n")
print(confusion_matrix(test_data.target, predicted))
