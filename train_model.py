import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_excel("dairy_predictive_maintenance_data.xlsx")

# Encode categorical variables
le_equipment = LabelEncoder()
df["Equipment_ID"] = le_equipment.fit_transform(df["Equipment_ID"])

le_failure = LabelEncoder()
df["Failure_Mode_Label"] = le_failure.fit_transform(df["Failure_Mode"])

# Define features and target
features = [
    "Equipment_ID", "SKU_Changeover", "Vibration_X", "Vibration_Y",
    "Vibration_Z", "Temperature", "Pressure", "Motor_Current", "Wear_Level"
]
X = df[features]
y = df["Failure_Mode_Label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_failure.classes_))

# Save model and encoders
joblib.dump(clf, "failure_predictor_model.pkl")
joblib.dump(le_equipment, "equipment_encoder.pkl")
joblib.dump(le_failure, "failure_encoder.pkl")

print("✅ Model and encoders saved!")
