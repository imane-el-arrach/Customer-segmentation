#Customer Segmentation with RFM Analysis

# Project Overview
This project implements customer segmentation analysis using online retail transaction data. The goal is to clean, explore, and prepare transactional data for RFM (Recency, Frequency, Monetary) analysis and subsequent clustering to identify customer segments.

# Dataset Information
The project uses the "Online Retail II" dataset from the UCI Machine Learning Repository, containing online retail transaction data.

# Dataset Source
Dataset: Online Retail II
Link: UCI Machine Learning Repository - Online Retail II
Format: Excel (.xlsx)
Size: ~46MB (for Year 2010-2011)

Note: The dataset is NOT included in this repository. You need to download it separately from the UCI link above and place it in the data/ directory.

# Notebook Workflow
1. Data Loading & Exploration
Load dataset from Excel file
Initial data exploration and summary statistics
Identify data quality issues

2. Data Cleaning
Filter valid transactions (regular invoices only)
Remove non-standard product codes (administrative, test items)
Handle missing Customer IDs
Remove negative/zero prices
Clean quantities and prices

3. Feature Engineering - RFM Analysis
Create RFM metrics for each customer:
MonetaryValue: Total purchase value (Quantity × Price)
Frequency: Number of unique invoices
Recency: Days since last purchase

4. Data Preparation for Clustering
Normalize RFM features using StandardScaler
Prepare data for K-means clustering

5. Clustering Analysis (Planned)
Determine optimal clusters (elbow method, silhouette score)
Apply K-means clustering
Analyze and interpret customer segments
Visualize clusters

## Setup & Environment

Create a python virtual environment via python -m venv <env_name>
Activate the environment via . <env_name>/Scripts/activate or . <env_name>/bin/activate
Install requirements via pip install -r requirements.txt
