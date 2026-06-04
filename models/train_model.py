import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/updated_version.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Target Column
target_column = "heart_attack"

# Features and Target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Save Model
joblib.dump(model, "models/model.pkl")

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

print("model.pkl saved successfully")
print("scaler.pkl saved successfully")
