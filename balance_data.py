"""
Data Balancing using SMOTE for Smartphone Price Prediction
Simplified version - SMOTE only
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading data...")
df = pd.read_csv('Data\Data_Preprocessed\test_processed.csv')

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
# SMOTE (Synthetic Minority Over-sampling Technique)
# ============================================================================
print("\n" + "="*70)
print("SMOTE - Synthetic Minority Over-sampling Technique")
print("="*70)
print("Creates synthetic samples for minority class using k-nearest neighbors")

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"\nNew shape: {X_smote.shape}")
print(f"New class distribution:\n{pd.Series(y_smote).value_counts()}")
print(f"Class 0: {(y_smote==0).sum()} ({(y_smote==0).sum()/len(y_smote)*100:.2f}%)")
print(f"Class 1: {(y_smote==1).sum()} ({(y_smote==1).sum()/len(y_smote)*100:.2f}%)")

# Save balanced data
df_smote = pd.concat([X_smote, pd.Series(y_smote, name='Price_Encoded')], axis=1)
df_smote.to_csv('train_balanced_smote.csv', index=False)
print("\n✓ Saved as 'train_balanced_smote.csv'")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*70)
print("Creating Visualization")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Before SMOTE
axes[0].bar(['Non-Expensive', 'Expensive'], 
            [y.value_counts()[0], y.value_counts()[1]],
            color=['lightgreen', 'lightcoral'])
axes[0].set_title('Before SMOTE\n(Imbalanced)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].set_ylim([0, max(y.value_counts()) * 1.2])
for i, v in enumerate([y.value_counts()[0], y.value_counts()[1]]):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Plot 2: After SMOTE
axes[1].bar(['Non-Expensive', 'Expensive'], 
            [pd.Series(y_smote).value_counts()[0], pd.Series(y_smote).value_counts()[1]],
            color=['lightgreen', 'lightcoral'])
axes[1].set_title('After SMOTE\n(Balanced)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Number of Samples')
axes[1].set_ylim([0, max(pd.Series(y_smote).value_counts()) * 1.2])
for i, v in enumerate([pd.Series(y_smote).value_counts()[0], pd.Series(y_smote).value_counts()[1]]):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('smote_balancing.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization as 'smote_balancing.png'")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Original samples: {len(y)}")
print(f"  - Non-Expensive: {(y==0).sum()}")
print(f"  - Expensive: {(y==1).sum()}")
print(f"  - Imbalance ratio: {(y==0).sum() / (y==1).sum():.2f}:1")
print(f"\nAfter SMOTE: {len(y_smote)}")
print(f"  - Non-Expensive: {(y_smote==0).sum()}")
print(f"  - Expensive: {(y_smote==1).sum()}")
print(f"  - Balanced: {(y_smote==0).sum() / (y_smote==1).sum():.2f}:1")
print(f"\nSynthetic samples created: {len(y_smote) - len(y)}")
print("="*70)
print("\n✅ SMOTE balancing completed successfully!")
