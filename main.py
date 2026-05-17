import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Features
X = data[['N','P','K','temperature',
          'humidity','ph','rainfall']]

# Label
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy*100,2), "%")

# User input
print("\nEnter values:")

N = float(input("Nitrogen (N): "))
P = float(input("Phosphorus (P): "))
K = float(input("Potassium (K): "))
temperature = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

# Prediction
input_data = pd.DataFrame(
    [[N,P,K,temperature,humidity,ph,rainfall]],
    columns=['N','P','K','temperature',
             'humidity','ph','rainfall']
)

prediction = model.predict(input_data)

print("\nRecommended Crop:", prediction[0])