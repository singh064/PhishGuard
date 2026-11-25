import json
import os

notebook_path = r'c:\Users\nisht\.gemini\antigravity\scratch\PhishGuard\notebooks\eda.ipynb'

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# PhishGuard – Stage 3: Data Cleaning & Exploratory Data Analysis (EDA)\n",
            "\n",
            "**Author:** PhishGuard AI Development Team  \n",
            "**Project:** Intelligent Phishing Website Detection System  \n",
            "**Dataset:** Phishing Site URLs (Kaggle / ODbL v1.0)  \n",
            "\n",
            "--- \n",
            "### Objectives:\n",
            "1. **Data Loading:** Ingest the raw real-world dataset and inspect its structure without assumptions.\n",
            "2. **Data Quality Audit:** Detect missing fields, duplicate rows, duplicate URLs, conflicting labels, and unnecessary columns.\n",
            "3. **Data Cleaning:** Cleanse exact duplicates and resolve label ambiguities while keeping the raw dataset untouched.\n",
            "4. **Exploratory Data Analysis (EDA):** Quantify class balance and analyze lexical properties like URL length distributions.\n",
            "5. **Insights for Machine Learning:** Formulate strategic conclusions for subsequent feature extraction and model training."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Setup & Environment\n",
            "Import essential data science libraries: Pandas, NumPy, Matplotlib, and Seaborn."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Visual configurations\n",
            "sns.set_theme(style='whitegrid', font='sans-serif')\n",
            "palette = {'good': '#2ecc71', 'bad': '#e74c3c'}\n",
            "\n",
            "# File paths\n",
            "RAW_DATA_PATH = os.path.join('..', 'data', 'phishing_site_urls.csv')\n",
            "CLEANED_DATA_PATH = os.path.join('..', 'data', 'cleaned_urls.csv')\n",
            "FIG_DIR = os.path.join('figures')\n",
            "os.makedirs(FIG_DIR, exist_ok=True)\n",
            "\n",
            "print('Environment initialized successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Task 1: Data Loading & Schema Inspection\n",
            "Loading the raw dataset and displaying its dimensions, column names, and data types."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load the raw CSV dataset\n",
            "raw_df = pd.read_csv(RAW_DATA_PATH)\n",
            "\n",
            "print('Dataset Dimensions:')\n",
            "print(f'Rows: {raw_df.shape[0]:,}')\n",
            "print(f'Columns: {raw_df.shape[1]}')\n",
            "print('\\nColumn Data Types & Non-Null Counts:')\n",
            "raw_df.info()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# First 5 records\n",
            "print('First 5 Records:')\n",
            "raw_df.head()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Last 5 records\n",
            "print('Last 5 Records:')\n",
            "raw_df.tail()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Schema Findings:\n",
            "- **URL Column:** `'URL'` (string object containing the raw web address).\n",
            "- **Target Label Column:** `'Label'` (string object containing categorical outcomes `'good'` or `'bad'`).\n",
            "- **Unnecessary Columns:** None. The dataset only contains the 2 essential columns required for URL feature extraction and binary classification."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Task 2: Data Quality Checking\n",
            "Auditing the integrity of the data across null values, duplicate entries, and label consistency."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Missing value check\n",
            "print('Missing Values per Column:')\n",
            "print(raw_df.isnull().sum())\n",
            "\n",
            "# 2. Exact duplicate rows (identical URL and identical Label)\n",
            "exact_duplicates = raw_df.duplicated().sum()\n",
            "print(f'\\nExact Duplicate Rows: {exact_duplicates:,}')\n",
            "\n",
            "# 3. Duplicate URLs\n",
            "url_duplicates = raw_df.duplicated(subset=['URL']).sum()\n",
            "unique_urls = raw_df['URL'].nunique()\n",
            "print(f'Total Duplicate URL Occurrences: {url_duplicates:,}')\n",
            "print(f'Unique URLs: {unique_urls:,}')\n",
            "\n",
            "# 4. Target label distribution in raw data\n",
            "print('\\nRaw Class Distribution:')\n",
            "print(raw_df['Label'].value_counts())\n",
            "print('\\nRaw Class Distribution (%):')\n",
            "print(raw_df['Label'].value_counts(normalize=True) * 100)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Check for Conflicting Labels (Same URL associated with different labels)\n",
            "conflicting_labels = raw_df.groupby('URL')['Label'].nunique()\n",
            "conflicting_urls = conflicting_labels[conflicting_labels > 1]\n",
            "print(f'Number of URLs with conflicting labels: {len(conflicting_urls)}')\n",
            "\n",
            "if len(conflicting_urls) > 0:\n",
            "    print('Conflicting URL instances:')\n",
            "    for url in conflicting_urls.index:\n",
            "        print(f'URL: {url} -> Labels: {raw_df[raw_df[\"URL\"] == url][\"Label\"].tolist()}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Task 3: Data Cleaning\n",
            "### Decisions and Rationales:\n",
            "1. **Zero Missing Values:** No imputation or row deletion needed for nulls.\n",
            "2. **Exact Duplicates Removal:** 42,150 identical rows removed to prevent training redundancy and evaluation leakage between splits.\n",
            "3. **Conflicting Label Resolution:** 1 domain (`tommyhumphreys.com/`) had contradictory labels (1 'good' record and 2 'bad' records). Both entries are dropped to avoid conflicting supervision signals.\n",
            "4. **Preserve Raw Dataset:** The raw `phishing_site_urls.csv` remains completely intact. The clean data is exported separately to `cleaned_urls.csv`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Remove exact duplicates\n",
            "clean_df = raw_df.drop_duplicates().copy()\n",
            "\n",
            "# Step 2: Remove conflicting duplicate URLs\n",
            "conflicting_mask = clean_df.duplicated(subset=['URL'], keep=False)\n",
            "clean_df = clean_df[~conflicting_mask].copy()\n",
            "\n",
            "# Step 3: Reset index cleanly\n",
            "clean_df = clean_df.reset_index(drop=True)\n",
            "\n",
            "# Step 4: Verify dimensions and distribution\n",
            "print(f'Cleaned Dataset Shape: {clean_df.shape[0]:,} rows, {clean_df.shape[1]} columns')\n",
            "print('\\nCleaned Class Distribution:')\n",
            "print(clean_df['Label'].value_counts())\n",
            "print('\\nCleaned Class Percentages:')\n",
            "print(clean_df['Label'].value_counts(normalize=True) * 100)\n",
            "\n",
            "# Step 5: Save cleaned dataset to separate CSV file\n",
            "clean_df.to_csv(CLEANED_DATA_PATH, index=False)\n",
            "print(f'\\nCleaned data successfully saved to: {CLEANED_DATA_PATH}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Task 4: Exploratory Data Analysis & Visualizations\n",
            "Visualizing class frequencies, proportions, and lexical length patterns."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Calculate URL string length for feature inspection\n",
            "clean_df['url_length'] = clean_df['URL'].str.len()\n",
            "\n",
            "# 1. Class Distribution Bar Chart\n",
            "plt.figure(figsize=(7, 5))\n",
            "ax = sns.countplot(x='Label', hue='Label', data=clean_df, palette=palette, order=['good', 'bad'], legend=False)\n",
            "plt.title('Distribution of Legitimate vs Phishing URLs', fontsize=14, fontweight='bold', pad=12)\n",
            "plt.xlabel('Website Classification', fontsize=12)\n",
            "plt.ylabel('Number of URLs', fontsize=12)\n",
            "plt.xticks([0, 1], ['Legitimate (Good)', 'Phishing (Bad)'], fontsize=11)\n",
            "\n",
            "for p in ax.patches:\n",
            "    height = int(p.get_height())\n",
            "    ax.annotate(f'{height:,}\\n({height/len(clean_df)*100:.1f}%)',\n",
            "                (p.get_x() + p.get_width() / 2., height / 2),\n",
            "                ha='center', va='center', fontsize=11, fontweight='bold', color='white')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIG_DIR, 'class_distribution_bar.png'), dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Class Proportion Pie Chart\n",
            "plt.figure(figsize=(6, 6))\n",
            "counts = clean_df['Label'].value_counts()\n",
            "plt.pie([counts['good'], counts['bad']],\n",
            "        labels=['Legitimate (Good)', 'Phishing (Bad)'],\n",
            "        autopct='%1.1f%%',\n",
            "        startangle=140,\n",
            "        colors=['#2ecc71', '#e74c3c'],\n",
            "        explode=(0, 0.08),\n",
            "        shadow=True,\n",
            "        textprops={'fontsize': 12, 'weight': 'bold'})\n",
            "plt.title('PhishGuard URL Class Breakdown', fontsize=14, fontweight='bold', pad=12)\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIG_DIR, 'class_distribution_pie.png'), dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Overall URL Length Statistical Summary\n",
            "print('Overall URL Length Statistics:')\n",
            "print(clean_df['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))\n",
            "\n",
            "p99 = clean_df['url_length'].quantile(0.99)\n",
            "plt.figure(figsize=(9, 5))\n",
            "sns.histplot(clean_df[clean_df['url_length'] <= p99]['url_length'], bins=50, kde=True, color='#3498db')\n",
            "plt.axvline(clean_df['url_length'].median(), color='#e67e22', linestyle='--', linewidth=2, label=f'Median: {clean_df[\"url_length\"].median():.0f}')\n",
            "plt.axvline(clean_df['url_length'].mean(), color='#9b59b6', linestyle='-', linewidth=2, label=f'Mean: {clean_df[\"url_length\"].mean():.1f}')\n",
            "plt.title('Overall URL Length Distribution (Up to 99th Percentile)', fontsize=14, fontweight='bold', pad=12)\n",
            "plt.xlabel('URL Length (characters)', fontsize=12)\n",
            "plt.ylabel('Frequency', fontsize=12)\n",
            "plt.legend(fontsize=11)\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIG_DIR, 'url_length_distribution.png'), dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Comparative URL Length: Boxplot & Density Estimation\n",
            "print('--- Legitimate URL Length Statistics ---')\n",
            "print(clean_df[clean_df['Label'] == 'good']['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))\n",
            "\n",
            "print('\\n--- Phishing URL Length Statistics ---')\n",
            "print(clean_df[clean_df['Label'] == 'bad']['url_length'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Boxplot\n",
            "sns.boxplot(ax=axes[0], x='Label', hue='Label', y='url_length', data=clean_df, palette=palette, order=['good', 'bad'], showfliers=False, legend=False)\n",
            "axes[0].set_title('URL Length Comparison (Boxplot - Outliers Filtered)', fontsize=13, fontweight='bold')\n",
            "axes[0].set_xticks([0, 1])\n",
            "axes[0].set_xticklabels(['Legitimate (Good)', 'Phishing (Bad)'], fontsize=11)\n",
            "axes[0].set_xlabel('Class', fontsize=12)\n",
            "axes[0].set_ylabel('URL Length (characters)', fontsize=12)\n",
            "\n",
            "# Comparative KDE\n",
            "sns.kdeplot(ax=axes[1], data=clean_df[(clean_df['url_length'] <= p99) & (clean_df['Label'] == 'good')]['url_length'], label='Legitimate (Good)', color='#2ecc71', fill=True, alpha=0.3)\n",
            "sns.kdeplot(ax=axes[1], data=clean_df[(clean_df['url_length'] <= p99) & (clean_df['Label'] == 'bad')]['url_length'], label='Phishing (Bad)', color='#e74c3c', fill=True, alpha=0.3)\n",
            "axes[1].set_title('URL Length Density Comparison (KDE)', fontsize=13, fontweight='bold')\n",
            "axes[1].set_xlabel('URL Length (characters)', fontsize=12)\n",
            "axes[1].set_ylabel('Density', fontsize=12)\n",
            "axes[1].legend(fontsize=11)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIG_DIR, 'url_length_comparison.png'), dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Key Insights & Machine Learning Implications\n",
            "\n",
            "1. **Imbalance (77.5% vs 22.5%):**\n",
            "   - While not critically extreme, phishing URLs represent approximately 1 out of every 4 URLs in the dataset.\n",
            "   - **ML Implication:** Plain accuracy will be deceptive (a naive model predicting all 'good' would achieve 77.5% accuracy). We must evaluate using Precision, Recall, F1-Score, and ROC-AUC, and apply stratified sampling (`stratify=y`) during train-test splitting.\n",
            "\n",
            "2. **URL Length as a Strong Predictive Signal:**\n",
            "   - Phishing URLs have a significantly higher mean length (71.06 characters vs 45.77 characters for legitimate sites).\n",
            "   - The 75th percentile for phishing is 82 characters, whereas legitimate sites have a 75th percentile of just 56 characters.\n",
            "   - **Reasoning:** Phishing URLs frequently incorporate lengthy obfuscated paths, subdomains impersonating real brands, and URL parameters to disguise destinations.\n",
            "\n",
            "3. **Ready for Feature Engineering (Stage 4):**\n",
            "   - The cleaned dataset (`cleaned_urls.csv`) contains 507,194 verified, non-redundant raw URLs, providing a solid foundation for lexical, structural, and security feature extraction."
        ]
    }
]

notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_dict, f, indent=2)

print("Notebook created successfully at:", notebook_path)
