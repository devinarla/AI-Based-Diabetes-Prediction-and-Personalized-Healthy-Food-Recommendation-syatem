import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Diabetes_AI/diabetes_ai_model_dataset_1000.CSV")

# Input features
X = data[
    [
        "Age",
        "Glucose",
        "BloodPressure",
        "BMI",
        "Insulin",
        "DiabetesPedigreeFunction",
    ]
]

# Output
y = data["Diabetes"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Diabetes AI Model Training Complete!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# User input
print("\nEnter patient details:")

age = float(input("Age: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
bmi = float(input("BMI: "))
insulin = float(input("Insulin: "))
pedigree = float(input("Diabetes Pedigree Function: "))

patient = pd.DataFrame(
    [[age, glucose, blood_pressure, bmi, insulin, pedigree]],
    columns=[
        "Age",
        "Glucose",
        "BloodPressure",
        "BMI",
        "Insulin",
        "DiabetesPedigreeFunction",
    ],
)

result = model.predict(patient)[0]

if result == 1:
    print("\nPrediction: Diabetic")
else:
    print("\nPrediction: Not Diabetic")