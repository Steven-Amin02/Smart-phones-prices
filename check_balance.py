import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
df = pd.read_csv(r'D:\collage\lv3\Sem1\Artificial intelligence_\Project\Smart Phone Prices Prediction\train_balanced_smote.csv')

# Check the distribution of the target variable (Price_Encoded)
print("=" * 50)
print("DATA BALANCE ANALYSIS")
print("=" * 50)
print("\n1. Class Distribution:")
print("-" * 50)
class_counts = df['Price_Encoded'].value_counts().sort_index()
print(class_counts)

print("\n2. Class Percentages:")
print("-" * 50)
class_percentages = df['Price_Encoded'].value_counts(normalize=True).sort_index() * 100
for class_label, percentage in class_percentages.items():
    print(f"Class {class_label}: {percentage:.2f}%")

print("\n3. Balance Assessment:")
print("-" * 50)
total_samples = len(df)
num_classes = df['Price_Encoded'].nunique()
expected_per_class = total_samples / num_classes

print(f"Total samples: {total_samples}")
print(f"Number of classes: {num_classes}")
print(f"Expected samples per class (if balanced): {expected_per_class:.2f}")

# Calculate imbalance ratio
max_class = class_counts.max()
min_class = class_counts.min()
imbalance_ratio = max_class / min_class

print(f"\nImbalance Ratio (max/min): {imbalance_ratio:.2f}")

if imbalance_ratio < 1.5:
    print("✓ Dataset is BALANCED")
elif imbalance_ratio < 3:
    print("⚠ Dataset is SLIGHTLY IMBALANCED")
else:
    print("✗ Dataset is HIGHLY IMBALANCED")

# Visualize the distribution
plt.figure(figsize=(12, 5))

# Bar plot
plt.subplot(1, 2, 1)
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Price Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, v in enumerate(class_counts):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(class_counts, labels=[f'Class {i}' for i in class_counts.index], 
        autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_balance_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'class_balance_visualization.png'")
plt.show()

print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
