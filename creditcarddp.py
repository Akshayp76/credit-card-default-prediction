import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def Credit_Card_Defaulter_Predictio_csv(file_path):
    data = pd.read_csv(file_path)
    return data

data = Credit_Card_Defaulter_Predictio_csv('credit_card_data.csv')


X = data.drop('default', axis=1)
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

rf_predictions = rf_classifier.predict(X_test_scaled)

print("Random Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train_scaled, y_train)

xgb_predictions = xgb_classifier.predict(X_test_scaled)

print("\nXGBoost Model Performance:")
print("Accuracy:", accuracy_score(y_test, xgb_predictions))
print("Classification Report:")
print(classification_report(y_test, xgb_predictions))