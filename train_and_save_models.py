# train_and_save_model.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

os.makedirs('static', exist_ok=True)

# Load dataset
df = pd.read_csv("Fitness_trackers.csv")

# Clean price columns
df['Selling Price'] = df['Selling Price'].astype(str).str.replace(',', '').replace('nan', pd.NA)
df['Selling Price'] = pd.to_numeric(df['Selling Price'], errors='coerce')
df['Original Price'] = df['Original Price'].astype(str).str.replace(',', '').replace('nan', pd.NA)
df['Original Price'] = pd.to_numeric(df['Original Price'], errors='coerce')

# Clean Reviews
df['Reviews'] = df['Reviews'].astype(str).str.replace(',', '').replace('nan', pd.NA)
df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median())

# Clean Battery Life
df['Average Battery Life (in days)'] = df['Average Battery Life (in days)'].astype(str).str.replace(',', '').replace('nan', pd.NA)
df['Average Battery Life (in days)'] = pd.to_numeric(df['Average Battery Life (in days)'], errors='coerce')
df['Average Battery Life (in days)'] = df['Average Battery Life (in days)'].fillna(df['Average Battery Life (in days)'].median())

# Create target: High_Rating (1 if Rating >= 4.0)
df['High_Rating'] = (df['Rating (Out of 5)'] >= 4.0).astype(int)

# Select features
features = ['Brand Name', 'Device Type', 'Selling Price', 'Original Price',
            'Average Battery Life (in days)', 'Reviews', 'Display', 'Strap Material']
X = df[features]
y = df['High_Rating']

# Encode categorical features
label_encoders = {}
for col in ['Brand Name', 'Device Type', 'Display', 'Strap Material']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Fill remaining NaN with median
X = X.fillna(X.median())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train pruned model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Save model & encoders
joblib.dump(model, 'best_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

# === Generate Required Visualizations ===

# 1. Results
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"Akurasi Latih: {train_acc:.4f}")
print(f"Akurasi Uji: {test_acc:.4f}")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Rendah (<4.0)', 'Tinggi (≥4.0)'],
            yticklabels=['Rendah (<4.0)', 'Tinggi (≥4.0)'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.savefig('static/confusion_matrix.png')
plt.close()

# 3. ROC & AUC
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='red', label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0,1], [0,1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig('static/roc_curve.png')
plt.close()

# 4. Feature Importance
importances = model.feature_importances_
feat_imp = pd.DataFrame({'Fitur': X.columns, 'Pentingnya': importances}).sort_values('Pentingnya', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x='Pentingnya', y='Fitur', data=feat_imp, palette='viridis')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# 5. Decision Tree
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=['Rendah', 'Tinggi'], filled=True, rounded=True)
plt.title("Decision Tree (max_depth=4)")
plt.savefig('static/decision_tree.png', bbox_inches='tight')
plt.close()

# 6. Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='High_Rating', data=df)
plt.title('Distribusi Kualitas Perangkat (0=Rendah, 1=Tinggi)')
plt.xlabel('High_Rating')
plt.ylabel('Jumlah')
plt.savefig('static/target_distribution.png')
plt.close()

print("✅ Model dan visualisasi berhasil disimpan!")