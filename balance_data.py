"""
Data Balancing Techniques for Smartphone Price Prediction
This script demonstrates multiple methods to balance an imbalanced dataset.
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading data...")
df = pd.read_csv('train_processed.csv')

# Separate features and target
X = df.drop('Price_Encoded', axis=1)
y = df['Price_Encoded']

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
if non_numeric_cols:
    print(f"\n⚠ Found non-numeric columns: {non_numeric_cols}")
    print("Converting to numeric or dropping...")
    
    # Try to convert to numeric, if fails, drop the column
    for col in non_numeric_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            print(f"  ✓ Converted {col} to numeric")
        except:
            print(f"  ✗ Dropping {col} (cannot convert)")
            X = X.drop(col, axis=1)
    
    # Fill any NaN values created during conversion
    X = X.fillna(X.mean())

print(f"\nOriginal dataset shape: {X.shape}")
print(f"Original class distribution:\n{y.value_counts()}")
print(f"Class 0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"Class 1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")

# ============================================================================
# METHOD 1: Random Oversampling (Duplicate minority class samples)
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: Random Oversampling")
print("="*70)
print("Duplicates minority class samples randomly until balanced")

ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X, y)

print(f"New shape: {X_ros.shape}")
print(f"New class distribution:\n{pd.Series(y_ros).value_counts()}")

# Save balanced data
df_ros = pd.concat([X_ros, pd.Series(y_ros, name='Price_Encoded')], axis=1)
df_ros.to_csv('train_balanced_oversampling.csv', index=False)
print("✓ Saved as 'train_balanced_oversampling.csv'")

# ============================================================================
# METHOD 2: Random Undersampling (Remove majority class samples)
# ============================================================================
print("\n" + "="*70)
print("METHOD 2: Random Undersampling")
print("="*70)
print("Removes majority class samples randomly until balanced")

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

print(f"New shape: {X_rus.shape}")
print(f"New class distribution:\n{pd.Series(y_rus).value_counts()}")

# Save balanced data
df_rus = pd.concat([X_rus, pd.Series(y_rus, name='Price_Encoded')], axis=1)
df_rus.to_csv('train_balanced_undersampling.csv', index=False)
print("✓ Saved as 'train_balanced_undersampling.csv'")

# ============================================================================
# METHOD 3: SMOTE (Synthetic Minority Over-sampling Technique)
# ============================================================================
print("\n" + "="*70)
print("METHOD 3: SMOTE (Recommended)")
print("="*70)
print("Creates synthetic samples for minority class using k-nearest neighbors")

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"New shape: {X_smote.shape}")
print(f"New class distribution:\n{pd.Series(y_smote).value_counts()}")

# Save balanced data
df_smote = pd.concat([X_smote, pd.Series(y_smote, name='Price_Encoded')], axis=1)
df_smote.to_csv('train_balanced_smote.csv', index=False)
print("✓ Saved as 'train_balanced_smote.csv'")

# ============================================================================
# METHOD 4: Combination (SMOTE + Tomek Links)
# ============================================================================
print("\n" + "="*70)
print("METHOD 4: SMOTE + Tomek Links")
print("="*70)
print("SMOTE to oversample, then Tomek links to clean borderline samples")

smt = SMOTETomek(random_state=42)
X_smt, y_smt = smt.fit_resample(X, y)

print(f"New shape: {X_smt.shape}")
print(f"New class distribution:\n{pd.Series(y_smt).value_counts()}")

# Save balanced data
df_smt = pd.concat([X_smt, pd.Series(y_smt, name='Price_Encoded')], axis=1)
df_smt.to_csv('train_balanced_smote_tomek.csv', index=False)
print("✓ Saved as 'train_balanced_smote_tomek.csv'")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*70)
print("Creating comparison visualization...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Data Balancing Techniques Comparison', fontsize=16, fontweight='bold')

methods = [
    ('Original', y),
    ('Random Oversampling', y_ros),
    ('Random Undersampling', y_rus),
    ('SMOTE', y_smote),
    ('SMOTE + Tomek', y_smt)
]

for idx, (name, data) in enumerate(methods):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    counts = pd.Series(data).value_counts().sort_index()
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Class', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Expensive', 'Expensive'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total = counts.sum()
    for i, (class_label, count) in enumerate(counts.items()):
        percentage = (count / total) * 100
        ax.text(i, count/2, f'{percentage:.1f}%',
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('balancing_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'balancing_comparison.png'")

# ============================================================================
# Summary and Recommendations
# ============================================================================
print("\n" + "="*70)
print("SUMMARY & RECOMMENDATIONS")
print("="*70)

print("""
📊 Files Created:
   1. train_balanced_oversampling.csv  - Random oversampling
   2. train_balanced_undersampling.csv - Random undersampling  
   3. train_balanced_smote.csv         - SMOTE (RECOMMENDED)
   4. train_balanced_smote_tomek.csv   - SMOTE + Tomek Links

💡 Which Method to Use?

   ✓ SMOTE (Method 3) - RECOMMENDED for your case
     - Creates synthetic samples, not just duplicates
     - Maintains data quality
     - Works well with moderate imbalance (2.56 ratio)
     - Best balance between performance and data size

   ⚠ Random Oversampling (Method 1)
     - Simple but may cause overfitting
     - Just duplicates existing samples
     - Use if SMOTE doesn't work well

   ⚠ Random Undersampling (Method 2)
     - Loses information (616 → 241 samples)
     - Only use if you have LOTS of data
     - NOT recommended for your dataset size

   ⚠ SMOTE + Tomek (Method 4)
     - More sophisticated, cleans noisy samples
     - Try if SMOTE alone doesn't improve results

📝 Next Steps:
   1. Use 'train_balanced_smote.csv' to retrain your model
   2. Compare performance with original data
   3. Check if F1-score and recall improve
   4. If results are worse, try class_weight parameter instead

🔧 Alternative: Use Class Weights (No resampling needed)
   In your XGBoost model, add:
   
   scale_pos_weight = len(y[y==0]) / len(y[y==1])  # ≈ 2.56
   
   model = XGBClassifier(scale_pos_weight=scale_pos_weight, ...)
   
   This tells the model to pay more attention to minority class!
""")

print("="*70)
print("BALANCING COMPLETE!")
print("="*70)
