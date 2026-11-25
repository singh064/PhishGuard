import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

cleaned_path = r'c:\Users\nisht\.gemini\antigravity\scratch\PhishGuard\data\cleaned_urls.csv'
fig_dir = r'c:\Users\nisht\.gemini\antigravity\scratch\PhishGuard\notebooks\figures'
os.makedirs(fig_dir, exist_ok=True)

print("Loading cleaned dataset...")
df = pd.read_csv(cleaned_path)
df['url_length'] = df['URL'].str.len()

print("=== OVERALL URL LENGTH SUMMARY ===")
print(df['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== GOOD (LEGITIMATE) URL LENGTH SUMMARY ===")
print(df[df['Label'] == 'good']['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

print("\n=== BAD (PHISHING) URL LENGTH SUMMARY ===")
print(df[df['Label'] == 'bad']['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# Set visual style
sns.set_theme(style='whitegrid', font='sans-serif')
palette = {'good': '#2ecc71', 'bad': '#e74c3c'}

# 1. Bar Chart - Class Counts
plt.figure(figsize=(7, 5))
ax = sns.countplot(x='Label', data=df, palette=palette, order=['good', 'bad'])
plt.title('Distribution of Legitimate vs Phishing URLs', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Website Classification', fontsize=12)
plt.ylabel('Number of URLs', fontsize=12)
plt.xticks([0, 1], ['Legitimate (Good)', 'Phishing (Bad)'], fontsize=11)
for p in ax.patches:
    height = int(p.get_height())
    ax.annotate(f'{height:,}\n({height/len(df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., height / 2),
                ha='center', va='center', fontsize=11, fontweight='bold', color='white')
plt.tight_layout()
bar_path = os.path.join(fig_dir, 'class_distribution_bar.png')
plt.savefig(bar_path, dpi=300)
plt.close()
print("Saved:", bar_path)

# 2. Pie Chart - Class Proportion
plt.figure(figsize=(6, 6))
counts = df['Label'].value_counts()
plt.pie([counts['good'], counts['bad']],
        labels=['Legitimate (Good)', 'Phishing (Bad)'],
        autopct='%1.1f%%',
        startangle=140,
        colors=['#2ecc71', '#e74c3c'],
        explode=(0, 0.08),
        shadow=True,
        textprops={'fontsize': 12, 'weight': 'bold'})
plt.title('PhishGuard URL Class Breakdown', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
pie_path = os.path.join(fig_dir, 'class_distribution_pie.png')
plt.savefig(pie_path, dpi=300)
plt.close()
print("Saved:", pie_path)

# 3. Overall URL Length Distribution
p99 = df['url_length'].quantile(0.99)
med_val = df['url_length'].median()
mean_val = df['url_length'].mean()

plt.figure(figsize=(9, 5))
sns.histplot(df[df['url_length'] <= p99]['url_length'], bins=50, kde=True, color='#3498db')
plt.axvline(med_val, color='#e67e22', linestyle='--', linewidth=2, label=f'Median: {med_val:.0f}')
plt.axvline(mean_val, color='#9b59b6', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.1f}')
plt.title('Overall URL Length Distribution (Up to 99th Percentile)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('URL Length (characters)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
dist_path = os.path.join(fig_dir, 'url_length_distribution.png')
plt.savefig(dist_path, dpi=300)
plt.close()
print("Saved:", dist_path)

# 4. Comparative URL Length: Boxplot & KDE Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot
sns.boxplot(ax=axes[0], x='Label', y='url_length', data=df, palette=palette, order=['good', 'bad'], showfliers=False)
axes[0].set_title('URL Length Comparison (Boxplot - Outliers Filtered)', fontsize=13, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Legitimate (Good)', 'Phishing (Bad)'], fontsize=11)
axes[0].set_xlabel('Class', fontsize=12)
axes[0].set_ylabel('URL Length (characters)', fontsize=12)

# Comparative KDE
sns.kdeplot(ax=axes[1], data=df[(df['url_length'] <= p99) & (df['Label'] == 'good')]['url_length'], label='Legitimate (Good)', color='#2ecc71', fill=True, alpha=0.3)
sns.kdeplot(ax=axes[1], data=df[(df['url_length'] <= p99) & (df['Label'] == 'bad')]['url_length'], label='Phishing (Bad)', color='#e74c3c', fill=True, alpha=0.3)
axes[1].set_title('URL Length Density Comparison (KDE)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('URL Length (characters)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].legend(fontsize=11)

plt.tight_layout()
comp_path = os.path.join(fig_dir, 'url_length_comparison.png')
plt.savefig(comp_path, dpi=300)
plt.close()
print("Saved:", comp_path)

print("EDA analysis script finished successfully!")
