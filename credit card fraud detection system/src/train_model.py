# task2/src/train_model.py
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pandas as pd
from preprocess import load_and_preprocess

def main():
    X, y, scaler = load_and_preprocess("data/creditcard.csv")
    print("Loaded shape:", X.shape)

    # quick sample option (uncomment in debug)
    # X = X.sample(20000, random_state=42)
    # y = y.loc[X.index]

    # Handle imbalance
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print("After resample:", X_res.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

    # Save model and scaler
    joblib.dump(model, "models/fraud_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

if __name__ == "__main__":
    main()
