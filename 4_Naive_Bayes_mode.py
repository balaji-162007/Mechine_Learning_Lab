import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Short dataset
data = pd.DataFrame({
    'Outlook': ['Sunny','Sunny','Overcast','Rainy','Rainy','Overcast'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal'],
    'Windy': ['False','True','False','False','False','True'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','Yes']
})

# Make explicit copy (fix warning)
X = data.iloc[:, :-1].copy()
y = data.iloc[:, -1]

# Encode categorical features
le = LabelEncoder()
for col in X.columns:
    X.loc[:, col] = le.fit_transform(X[col])   # safe assignment

y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Train
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
