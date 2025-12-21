import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set_theme(style="whitegrid")

# Load data
try:
    df = pd.read_csv('train_processed.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: train_processed.csv not found.")
    exit()

# Ensure images directory exists
if not os.path.exists('images'):
    os.makedirs('images')

# --- 1. Correlation Analysis ---
print("\n--- Correlation Analysis ---")
# Select numerical columns of interest
cols_of_interest = [
    'Price_Encoded', 'RAM Size GB', 'battery_capacity', 
    'Resolution_Width', 'Resolution_Height', 'fast_charging_power',
    'Screen_Size', 'Refresh_Rate', 'rating', 'Clock_Speed_GHz'
]

# Calculate correlation matrix
corr_matrix = df[cols_of_interest].corr()

# Get top correlations with Price
price_corr = corr_matrix['Price_Encoded'].sort_values(ascending=False)
print("\nTop Correlations with Price (Expensive=1):")
print(price_corr)

# Generate Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png')
print("Saved images/correlation_heatmap.png")
plt.close()

# --- 2. Feature Distributions vs Price ---

# Helper to plot boxplots
def plot_boxplot(x, y, title, filename):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df) # Removed custom palette to avoid TypeErrors
    plt.title(title)
    plt.xlabel('Price Category (0=Non-Expensive, 1=Expensive)')
    plt.ylabel(y)
    plt.xticks([0, 1], ['Non-Expensive', 'Expensive'])
    plt.tight_layout()
    plt.savefig(f'images/{filename}')
    print(f"Saved images/{filename}")
    plt.close()

# Plot key features
# RAM (Normalized) vs Price
plot_boxplot('Price_Encoded', 'RAM Size GB', 'RAM Distribution by Price Category', 'ram_vs_price.png')

# Battery (Normalized) vs Price
plot_boxplot('Price_Encoded', 'battery_capacity', 'Battery Capacity by Price Category', 'battery_vs_price.png')

# Resolution Width vs Price
plot_boxplot('Price_Encoded', 'Resolution_Width', 'Screen Resolution (Width) by Price Category', 'resolution_vs_price.png')

# Fast Charging vs Price
plot_boxplot('Price_Encoded', 'fast_charging_power', 'Fast Charging Speed by Price Category', 'charging_vs_price.png')

print("\nAnalysis Complete.")
