# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Load your historical customer data (replace 'telecom_churn.csv' with your dataset)
data = pd.read_csv('telecom_churn.csv')

# Data Preprocessing
# Handle missing values (if any)
data.dropna(inplace=True)

# Automatically identify and label encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Feature Selection
# Identify relevant features that might affect churn based on domain knowledge or feature importance analysis.

# Data Splitting
X = data.drop(columns=['Churn'])  # Features
y = data['Churn']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (if needed)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building
# 1. Logistic Regression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# 2. Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# 3. Gradient Boosting
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train, y_train)

# Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

print("Logistic Regression Model:")
evaluate_model(logistic_model, X_test, y_test)

print("Random Forest Model:")
evaluate_model(random_forest_model, X_test, y_test)

print("Gradient Boosting Model:")
evaluate_model(gradient_boosting_model, X_test, y_test)

# Model Tuning (Fine-tuning hyperparameters)
# Perform hyperparameter tuning on the best-performing model (e.g., Random Forest or Gradient Boosting) to optimize performance.

# Deployment
# Once the model is fine-tuned and meets your performance criteria, deploy it for predicting customer churn in production.

# Example of deploying a model with Flask or other deployment frameworks.

# You can also periodically retrain your model with new data to improve its accuracy and adapt to changing customer behavior.

