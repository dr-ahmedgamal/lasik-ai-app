import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the updated dataset
df = pd.read_csv("patients.csv")

# Feature columns and targets
X = df[["Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "Age"]]
y_reg = df[["Outcome_BCVA", "Outcome_K1", "Outcome_K2", "Outcome_Thickness"]]
y_class = df["Best_Surgery"]

# Encode the class labels
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)

# Train regression model
reg_model = RandomForestRegressor()
reg_model.fit(X, y_reg)

# Train classification model
class_model = RandomForestClassifier()
class_model.fit(X, y_class_encoded)

# Save models and label encoder
joblib.dump(reg_model, "model.pkl")
joblib.dump(class_model, "surgery_model.pkl")
joblib.dump(le, "label_encoder.pkl")