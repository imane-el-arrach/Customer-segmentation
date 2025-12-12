import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Style CSS

st.markdown("""
<style>
.main-title { font-size:2.5rem; font-weight:bold; text-align:center; background: linear-gradient(90deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:1rem;}
.section-title { font-size:1.8rem; color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:10px; margin-top:2rem; margin-bottom:1.5rem;}
.metric-box {background:#f8f9fa; border-radius:10px; padding:15px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.05);}
.insight-box {background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; border-radius:10px; padding:20px; margin:10px 0;}
.cluster-card {background:white; border-radius:10px; padding:20px; margin:10px 0; box-shadow:0 4px 6px rgba(0,0,0,0.1); border-left:5px solid;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">👥 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive dashboard with RFM analysis & KMeans clustering")


cluster_labels = {
    0: "RETAIN",
    1: "RE-ENGAGE",
    2: "NURTURE",
    3: "REWARD",
    -1: "PAMPER",
    -2: "UPSELL",
    -3: "DELIGHT"
}



# Load / Generate Data

@st.cache_data
def load_data():
    np.random.seed(42)
    n_customers = 2000
    n_outliers = 90

    # Non-outliers
    df_normal = pd.DataFrame({
        'CustomerID': range(1000, 1000+n_customers),
        'Recency': np.random.gamma(2,30,n_customers),
        'Frequency': np.random.poisson(3,n_customers),
        'MonetaryValue': np.random.exponential(100,n_customers)
    })

    # Outliers
    df_out = pd.DataFrame({
        'CustomerID': range(10000,10000+n_outliers),
        'Recency': np.random.gamma(2,100,n_outliers),
        'Frequency': np.random.poisson(10,n_outliers),
        'MonetaryValue': np.random.exponential(500,n_outliers)
    })

    # Assign outlier clusters
    out_clusters = [-1,-2,-3]
    df_out['Cluster'] = [out_clusters[i%3] for i in range(n_outliers)]

    # KMeans on normal data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_normal[['Recency','Frequency','MonetaryValue']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_normal['Cluster'] = kmeans.fit_predict(X_scaled)

    # Customer Value
    df_normal['Customer_Value'] = df_normal['Frequency'] * df_normal['MonetaryValue']
    df_out['Customer_Value'] = df_out['Frequency'] * df_out['MonetaryValue']

    # Merge
    df = pd.concat([df_normal, df_out], ignore_index=True)

    # Map cluster labels
    df['Segment'] = df['Cluster'].map(cluster_labels)

    return df

df = load_data()


# Sidebar

with st.sidebar:
    st.markdown("## ⚙️ Filters & Options")
    segments_all = df['Segment'].unique()
    selected_segments = st.multiselect("Select segments to display", options=segments_all, default=list(segments_all))
    min_val, max_val = st.slider("Filter by Monetary Value", float(df['MonetaryValue'].min()), float(df['MonetaryValue'].max()), (float(df['MonetaryValue'].min()), float(df['MonetaryValue'].max())))
    show_3d = st.checkbox("Show 3D Plot", value=True)

# Filter data
filtered_df = df[df['Segment'].isin(selected_segments)]
filtered_df = filtered_df[(filtered_df['MonetaryValue']>=min_val) & (filtered_df['MonetaryValue']<=max_val)]


# Overview Metrics

st.markdown('<h2 class="section-title">📋 Overview</h2>', unsafe_allow_html=True)
col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Customers", len(filtered_df))
col2.metric("Average Value", f"${filtered_df['MonetaryValue'].mean():,.0f}")
col3.metric("High Value Customers", len(filtered_df[filtered_df['MonetaryValue']>filtered_df['MonetaryValue'].quantile(0.8)]))
col4.metric("Recently Active", len(filtered_df[filtered_df['Recency']<30]))

# Distribution segments
if len(filtered_df)==0:
    st.warning("⚠️ No segment selected or filter too strict.")
else:
    fig = px.pie(filtered_df, names='Segment', values='CustomerID', title="Segment Distribution", color='Segment', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig,use_container_width=True)


# Profile per Cluster

st.markdown('<h2 class="section-title">👤 Segment Profiles</h2>', unsafe_allow_html=True)
segments_unique = filtered_df['Segment'].unique()
if len(segments_unique)==0:
    st.info("Select at least one segment to display profiles.")
else:
    tabs = st.tabs([f"{s}" for s in segments_unique])
    for i, seg in enumerate(segments_unique):
        with tabs[i]:
            seg_df = filtered_df[filtered_df['Segment']==seg]
            st.metric("Number of Customers", len(seg_df))
            st.metric("Average Value", f"${seg_df['MonetaryValue'].mean():,.0f}")
            st.metric("Total Value", f"${seg_df['Customer_Value'].sum():,.0f}")

            fig = make_subplots(rows=1,cols=3,subplot_titles=['Recency','Frequency','MonetaryValue'])
            fig.add_trace(go.Histogram(x=seg_df['Recency'],name='Recency'),row=1,col=1)
            fig.add_trace(go.Histogram(x=seg_df['Frequency'],name='Frequency'),row=1,col=2)
            fig.add_trace(go.Histogram(x=seg_df['MonetaryValue'],name='MonetaryValue'),row=1,col=3)
            fig.update_layout(height=400,showlegend=False)
            st.plotly_chart(fig,use_container_width=True)

            st.markdown("###  Top 5 Customers")
            top5 = seg_df.nlargest(5,'Customer_Value')[['CustomerID','Recency','Frequency','MonetaryValue','Customer_Value']]
            st.dataframe(top5.style.format({'MonetaryValue':'${:.2f}','Customer_Value':'${:.2f}'}),use_container_width=True)
            # Insights & Recommendations automatiques
            
            st.markdown("###  Insights & Recommendations")
            insights = []
            # Recency thresholds
            rec_med = seg_df['Recency'].median()
            freq_med = seg_df['Frequency'].median()
            mon_med = seg_df['MonetaryValue'].median()

            if seg in ['RETAIN','REWARD','PAMPER']:
                insights.append(" High value & frequent customers: reward loyalty, offer VIP perks, exclusive campaigns.")
            if seg in ['RE-ENGAGE','CHURN','UPSELL']:
                insights.append(" Low engagement: send re-engagement campaigns, discounts, targeted communication.")
            if seg in ['NURTURE','DELIGHT']:
                insights.append(" Encourage repeat purchases: bundles, cross-sell recommendations, loyalty points.")
            
            if rec_med>50:
                insights.append(" Customers have been inactive recently: consider reminder emails.")
            if freq_med<2:
                insights.append(" Low purchase frequency: suggest attractive starter offers.")
            if mon_med>300:
                insights.append(" High monetary value: focus on premium products and upselling.")

            st.markdown("• " + "\n• ".join(insights))

# 3D RFM Visualization

st.markdown('<h2 class="section-title">📈 RFM 3D Visualization</h2>', unsafe_allow_html=True)
if show_3d and len(filtered_df)>0:
    fig3d = px.scatter_3d(filtered_df, x='Recency', y='Frequency', z='MonetaryValue',
                        color='Segment', size='Customer_Value', hover_data=['CustomerID'],
                        color_discrete_sequence=px.colors.qualitative.Set2)
    fig3d.update_layout(scene=dict(xaxis_title='Recency',yaxis_title='Frequency',zaxis_title='MonetaryValue'),height=700)
    st.plotly_chart(fig3d,use_container_width=True)


# Pareto Analysis

if len(filtered_df)>0:
    df_sorted = filtered_df.sort_values('Customer_Value',ascending=False)
    df_sorted['Cumulative_Value'] = df_sorted['Customer_Value'].cumsum()
    df_sorted['Cumulative_Percent'] = (np.arange(1,len(df_sorted)+1)/len(df_sorted))*100
    df_sorted['Value_Percent'] = (df_sorted['Cumulative_Value']/df_sorted['Customer_Value'].sum())*100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['Cumulative_Percent'],y=df_sorted['Value_Percent'],mode='lines',name='Cumulative Value',line=dict(color='#3498db',width=3)))
    fig.add_shape(type="line",x0=20,y0=0,x1=20,y1=80,line=dict(color="red",width=2,dash="dash"))
    fig.add_shape(type="line",x0=0,y0=80,x1=20,y1=80,line=dict(color="red",width=2,dash="dash"))
    fig.update_layout(title="Pareto Analysis (20% customers ≈ 80% value)", xaxis_title="% Customers", yaxis_title="% Value", height=500)
    st.plotly_chart(fig,use_container_width=True)


# Export

st.markdown('<h2 class="section-title">📤 Export Data</h2>', unsafe_allow_html=True)
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(" Download Filtered Segments",data=csv,file_name="segments_clients.csv",mime="text/csv")
