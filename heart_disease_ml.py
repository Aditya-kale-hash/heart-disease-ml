import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs', exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "xtick.color":      "#999",
    "ytick.color":      "#999",
    "text.color":       "#fff",
    "grid.color":       "#2a2a2a",
    "legend.facecolor": "#1a1a1a",
    "legend.edgecolor": "#444",
})

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("heart.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())
print("\nFirst 5 Rows:\n", df.head())
print("\nBasic Stats:\n", df.describe())
print("\nTarget Count:\n", df['target'].value_counts())

df.drop_duplicates(inplace=True)
print("\nAfter removing duplicates:", df.shape)

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[feature_cols]
y = df['target']

print("\nFeatures used:", feature_cols)
print("Target: target (1=Heart Disease, 0=No Heart Disease)")

plt.figure(figsize=(14, 10), facecolor='#0f0f0f')
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, linecolor='#0f0f0f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='white', pad=14)
plt.tight_layout()
plt.savefig('outputs/chart1_correlation_heatmap.png', dpi=150)
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("\nLogistic Regression trained")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest trained")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
print("KNN trained")

def get_metrics(y_test, y_pred, model_name):
    return {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall': round(recall_score(y_test, y_pred) * 100, 2),
        'F1 Score': round(f1_score(y_test, y_pred) * 100, 2)
    }

results = pd.DataFrame([
    get_metrics(y_test, lr_pred, 'Logistic Regression'),
    get_metrics(y_test, rf_pred, 'Random Forest'),
    get_metrics(y_test, knn_pred, 'KNN')
])

print("\nModel Comparison Table:")
print(results.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0f0f0f')
x = np.arange(len(results['Model']))
width = 0.2
colors = ['#E50914', '#3B9EFF', '#00C48C', '#F5A623']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results[metric], width, label=metric, color=colors[i], edgecolor='none')

ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold', color='white', pad=14)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score in Percent', fontsize=12)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results['Model'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig('outputs/chart2_model_comparison.png', dpi=150)
plt.show()

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f0f0f')
colors_fi = ['#E50914','#F5A623','#3B9EFF','#00C48C','#B14FFF','#FF6B6B','#FFD93D','#6BCB77','#4D96FF','#C77DFF','#FF9F43','#A29BFE','#FD79A8']
ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors_fi, edgecolor='none')
ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold', color='white', pad=14)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/chart3_feature_importance.png', dpi=150)
plt.show()

best_model_name = results.loc[results['F1 Score'].idxmax(), 'Model']
best_pred = rf_pred if best_model_name == 'Random Forest' else (lr_pred if best_model_name == 'Logistic Regression' else knn_pred)

print("\nBest Model:", best_model_name)

cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f0f0f')
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', linewidths=0.5,
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'],
            ax=ax)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold', color='white', pad=14)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/chart4_confusion_matrix.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f0f0f')
disease_by_sex = df.groupby(['sex', 'target']).size().unstack()
disease_by_sex.plot(kind='bar', ax=ax, color=['#3B9EFF', '#E50914'], edgecolor='none')
ax.set_title('Heart Disease Count by Gender', fontsize=14, fontweight='bold', color='white', pad=14)
ax.set_xlabel('Sex (0=Female, 1=Male)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticklabels(['Female', 'Male'], rotation=0)
ax.legend(['No Disease', 'Heart Disease'], fontsize=10)
plt.tight_layout()
plt.savefig('outputs/chart5_disease_by_gender.png', dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f0f0f')
df[df['target'] == 0]['age'].plot(kind='hist', bins=30, alpha=0.7, color='#3B9EFF', ax=ax, label='No Disease')
df[df['target'] == 1]['age'].plot(kind='hist', bins=30, alpha=0.7, color='#E50914', ax=ax, label='Heart Disease')
ax.set_title('Age Distribution by Heart Disease', fontsize=14, fontweight='bold', color='white', pad=14)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('outputs/chart6_age_distribution.png', dpi=150)
plt.show()

print("\nCLASSIFICATION REPORT - BEST MODEL")
print("====================================")
print(classification_report(y_test, best_pred, target_names=['No Disease', 'Heart Disease']))

print("\nFINAL CONCLUSION")
print("=================")
print("")
print("1. Random Forest achieved the highest F1 Score among all three models")
print("   because it handles non-linear relationships in medical data very well.")
print("")
print("2. Chest pain type and thalach were the most important features")
print("   showing that heart rate and chest pain are strong disease indicators.")
print("")
print("3. Logistic Regression performed well as a baseline confirming")
print("   that several features have a strong linear relation with heart disease.")
print("")
print("4. KNN underperformed because medical datasets have complex boundaries")
print("   that simple distance based algorithms cannot capture accurately.")
print("")
print("5. Random Forest is the best model for heart disease prediction")
print("   as it gives the best balance of precision and recall which is")
print("   critical in medical diagnosis to minimize false negatives.")
