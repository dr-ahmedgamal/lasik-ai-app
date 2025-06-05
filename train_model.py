import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("patients.csv")

# Define features and targets
X = df[["Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "Age"]]
y_reg = df[["Outcome_BCVA", "Outcome_K1", "Outcome_K2", "Outcome_Thickness"]]
y_class = df["Best_Surgery"]

# Encode target labels for classification
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# Split data
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_class_train, y_class_test = train_test_split(X, y_class_encoded, test_size=0.2, random_state=42)

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_reg_train)

# Train classification model
class_model = RandomForestClassifier(n_estimators=100, random_state=42)
class_model.fit(X_train, y_class_train)

# Save models
joblib.dump(reg_model, "model.pkl")
joblib.dump(class_model, "surgery_model.pkl")
joblib.dump(le, "label_encoder.pkl")
