# Customer Segmentation with RFM Analysis & Streamlit App

## Project Overview
This project implements customer segmentation using online retail transaction data. It includes data cleaning, RFM (Recency, Frequency, Monetary) analysis, K-means clustering, outlier handling, and an interactive **Streamlit dashboard** for visualization and exploration of customer segments.

## Dataset
- **Source:** Online Retail II dataset (UCI Machine Learning Repository)
- **Format:** Excel (.xlsx)
- **Size:** ~46MB
- **Note:** Dataset is **not included** in this repository. Download it and place it in the `data/` directory.

## Notebook Workflow
1. **Data Loading & Exploration**
   - Load dataset from Excel
   - Initial exploration and summary statistics
   - Identify data quality issues

2. **Data Cleaning**
   - Filter valid transactions
   - Remove non-standard product codes
   - Handle missing Customer IDs
   - Remove negative/zero prices

3. **Feature Engineering - RFM**
   - **Recency:** Days since last purchase
   - **Frequency:** Number of purchases
   - **MonetaryValue:** Total spend (Quantity × Price)

4. **Outlier Handling**
   - Identify and separate outliers for analysis
   - Ensure clustering is robust to extreme values

5. **Clustering Analysis**
   - Normalize RFM features with StandardScaler
   - Apply **K-means clustering**
   - Analyze, interpret, and label customer segments
   - Visualize clusters interactively

6. **Streamlit Dashboard**
   - Interactive visualization of clusters
   - View metrics per segment
   - Filter by outliers/non-outliers
   - Download CSV for each segment
   - Insights and recommendations for marketing actions

## Setup & Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

