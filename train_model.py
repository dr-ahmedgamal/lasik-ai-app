import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save_models(data_path="patients.csv"):
    # Load updated patient data
    df = pd.read_csv(data_path)
    
    # Define feature columns
    features = ["Spherical", "Cylinder", "K1", "K2", "Pachymetry", "Pre-op BCVA", "Age"]
    
    # Regression targets: post-op BCVA, K1, K2, Corneal Thickness
    targets_reg = ["Outcome_BCVA", "Outcome_K1", "Outcome_K2", "Outcome_Thickness"]
    
    # Classification target: best surgery option
    target_clf = "Best_Surgery"
    
    # Prepare regression data
    X_reg = df[features]
    y_reg = df[targets_reg]
    
    # Prepare classification data
    X_clf = df[features]
    y_clf = df[target_clf]
    
    # Encode classification labels
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)
    
    # Train regression model (multi-output)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_reg, y_reg)
    
    # Train classification model
    class_model = RandomForestClassifier(n_estimators=100, random_state=42)
    class_model.fit(X_clf, y_clf_enc)
    
    # Save models and label encoder
    joblib.dump(reg_model, "model.pkl")
    joblib.dump(class_model, "surgery_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    
    print("Models and encoder saved successfully.")

if __name__ == "__main__":
    train_and_save_models()
