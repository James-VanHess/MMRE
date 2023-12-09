# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve

# Load car evaluation dataset
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car.data', header=None, names=col_names)

# Display basic information about the dataset
df_info = df.info()

# Explore unique values in feature columns and the label column
for feature in df.columns[:-1]:
    unique_vals = df[feature].unique()
    print(f"{feature}: {len(unique_vals)} values, {unique_vals}")
    
label_name = df.columns[-1]
label_unique_vals = df[label_name].unique()
print(f"{label_name}: {len(label_unique_vals)} values, {label_unique_vals}")

# Display total counts for each unique value in the label column
label_value_counts = df[label_name].value_counts()
print(label_value_counts)

# Encode ordinal data and create dummy columns
ordinal_mapping = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2},
}

df_mapped = df.replace(ordinal_mapping)
print(df_mapped)
df_encoded_mapped = pd.get_dummies(df_mapped, columns=col_names[:-1], drop_first=True)
df_encoded_mapped['class'], class_uniques_mapped = pd.factorize(df_encoded_mapped['class'])
class_col_mapped = df_encoded_mapped['class']
df_encoded_mapped.drop(columns=['class'], inplace=True)
last_col_pos_mapped = df_encoded_mapped.columns.get_loc('safety_2') + 1
df_encoded_mapped.insert(last_col_pos_mapped, 'class', class_col_mapped)
print(df_encoded_mapped)

# Split data into features (X) and labels (y)
X_mapped = df_encoded_mapped.loc[:, 'buying_1':'safety_2']
y_mapped = df_encoded_mapped['class']
#print(X_mapped)
#print(y_mapped)

# Split data into training and test sets
X_train_mapped, X_test_mapped, y_train_mapped, y_test_mapped = train_test_split(X_mapped, y_mapped, test_size=0.25, random_state=42)

# Create and train a random forest classifier with ordinality
clf_mapped = DecisionTreeClassifier(random_state=42)
clf_mapped.fit(X_train_mapped, y_train_mapped)

# Predictions on the test set
y_pred_mapped = clf_mapped.predict(X_test_mapped)
y_pred_mapped[0:5]

class_uniques_mapped

# Evaluate the model
accuracy_mapped = accuracy_score(y_test_mapped, y_pred_mapped)
conf_matrix_mapped = confusion_matrix(y_test_mapped, y_pred_mapped)
class_report_mapped = classification_report(y_test_mapped, y_pred_mapped)

# Feature selection using importance-based selection
feature_selector = SelectFromModel(clf_mapped, prefit=True)
X_train_selected = feature_selector.transform(X_train_mapped)
X_test_selected = feature_selector.transform(X_test_mapped)

# Create and train a new decision tree classifier with selected features
clf_selected = DecisionTreeClassifier(random_state=42)
clf_selected.fit(X_train_selected, y_train_mapped)

# Predictions on the test set with selected features
y_pred_selected = clf_selected.predict(X_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(y_test_mapped, y_pred_selected)

# Display results
print("Original Dataset Information:")
print(df_info)
print("\nEncoded Dataset Information:")
print(df_encoded_mapped.info())
print("\nOrdinality Leveraging Model Accuracy:")
print(f"Accuracy: {accuracy_mapped:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_mapped)
print("\nClassification Report:")
print(class_report_mapped)
print("\nFeature Selection Model Accuracy:")
print(f"Accuracy with selected features: {accuracy_selected:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_mapped, annot=True, fmt='d', cmap='Blues', xticklabels=class_uniques_mapped, yticklabels=class_uniques_mapped)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display the decision tree
import numpy as np
from sklearn.tree import plot_tree

# Visualize the decision tree without GraphViz
plt.figure(figsize=(20,10))
plot_tree(clf_mapped, feature_names=X_mapped.columns, class_names=class_uniques_mapped, filled=True, rounded=True, fontsize=10)
plt.show()

# K-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = cross_val_score(clf_mapped, X_mapped, y_mapped, cv=cv, scoring='accuracy')

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(clf_mapped, X_mapped, y_mapped, cv=cv, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Feature Importance
feature_importance = clf_mapped.feature_importances_
feature_names = X_mapped.columns
df_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_importance, palette='viridis')
plt.title('Feature Importance')
plt.show()


# Plot misclassifications vs. model confidence
proba_correct = clf_mapped.predict_proba(X_test_mapped).max(axis=1)
df_proba = pd.DataFrame({'Correct': (y_pred_mapped == y_test_mapped), 'Confidence': proba_correct})

df_proba.duplicated('Confidence').any()
df_proba = df_proba.drop_duplicates('Confidence')
df_proba.duplicated('Confidence').any()
df_proba['Confidence'].var()
df_proba['Confidence'] += np.random.normal(0, 0.001, size=len(df_proba))
df_proba['Confidence'].var()

plt.figure(figsize=(8, 6))
sns.histplot(data=df_proba, x='Confidence', hue='Correct', bins=20, kde=True, palette='coolwarm')
plt.title('Misclassifications vs. Model Confidence')
plt.xlabel('Model Confidence')
plt.ylabel('Count')
plt.show()