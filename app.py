import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Sales Forecast Dashboard",
                   page_icon="📊",
                   layout="wide")

# ==========================
# Load Dataset
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"],dayfirst=True)
    return df

df = load_data()

# ==========================
# Sidebar
# ==========================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segments"
    ]
)

# ==========================
# PAGE 1
# ==========================
if page == "Sales Overview":

    st.title("📊 Sales Overview Dashboard")

    total_sales = df["Sales"].sum()
    total_orders = len(df)
    avg_order = df["Sales"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Sales", f"${total_sales:,.2f}")
    c2.metric("Total Orders", total_orders)
    c3.metric("Average Order Value", f"${avg_order:,.2f}")

    st.divider()

    region = st.sidebar.multiselect(
        "Select Region",
        df["Region"].unique(),
        default=df["Region"].unique()
    )

    category = st.sidebar.multiselect(
        "Select Category",
        df["Category"].unique(),
        default=df["Category"].unique()
    )

    filtered = df[
        (df["Region"].isin(region)) &
        (df["Category"].isin(category))
    ]

    # Monthly Sales
    monthly = (
        filtered
        .groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Yearly Sales
    filtered["Year"] = filtered["Order Date"].dt.year

    yearly = (
        filtered
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        yearly,
        x="Year",
        y="Sales",
        title="Total Sales by Year",
        color="Sales"
    )

    st.plotly_chart(fig2, use_container_width=True)

# ==========================
# PAGE 2
# ==========================
elif page == "Forecast Explorer":

    st.title("📈 Forecast Explorer")

    option = st.selectbox(
        "Select Forecast",
        [
            "Furniture",
            "Technology",
            "Office Supplies",
            "East",
            "West"
        ]
    )

    horizon = st.slider(
        "Forecast Horizon",
        1,
        3,
        3
    )

    st.subheader(f"Forecast for {option}")

    # Example values
    if option == "Furniture":
        forecast = [28505, 28480, 28430]

    elif option == "Technology":
        forecast = [26000, 25980, 25950]

    elif option == "Office Supplies":
        forecast = [31480, 31420, 31380]

    elif option == "East":
        forecast = [31390, 31340, 31280]

    else:
        forecast = [27080, 27040, 27000]

    months = ["Month 1", "Month 2", "Month 3"]

    forecast_df = pd.DataFrame({
        "Month": months[:horizon],
        "Forecast": forecast[:horizon]
    })

    fig3 = px.line(
        forecast_df,
        x="Month",
        y="Forecast",
        markers=True,
        title="Sales Forecast"
    )

    st.plotly_chart(fig3, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "18,556.18")
    c2.metric("RMSE", "20,500.22")
    c3.metric("MAPE", "19.01 %")

    st.dataframe(forecast_df)

# ==========================
# PAGE 3
# ==========================
elif page == "Anomaly Report":

    st.title("🚨 Sales Anomaly Report")

    from sklearn.ensemble import IsolationForest

    weekly_sales = (
        df.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
        .sum()
        .reset_index()
    )

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = model.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomaly = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    fig = px.line(
        weekly_sales,
        x="Order Date",
        y="Sales",
        title="Weekly Sales with Anomalies"
    )

    fig.add_scatter(
        x=anomaly["Order Date"],
        y=anomaly["Sales"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="Anomaly"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detected Anomalies")

    st.dataframe(anomaly)

    csv = anomaly.to_csv(index=False).encode()

    st.download_button(
        "Download Anomaly Report",
        csv,
        "anomalies.csv",
        "text/csv"
    )

    # ==========================
# PAGE 4
# ==========================
elif page == "Demand Segments":

    st.title("📦 Product Demand Segments")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Aggregate Features
    subcat = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales","sum"),
        Average_Order_Value=("Sales","mean")
    ).reset_index()

    monthly = (
        df.groupby(
            [
                "Sub-Category",
                pd.Grouper(key="Order Date",freq="M")
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    volatility = (
        monthly.groupby("Sub-Category")["Sales"]
        .std()
        .reset_index()
    )

    volatility.columns = [
        "Sub-Category",
        "Sales_Volatility"
    ]

    yearly = (
        df.groupby(
            [
                "Sub-Category",
                pd.Grouper(key="Order Date",freq="Y")
            ]
        )["Sales"]
        .sum()
        .reset_index()
    )

    yearly["Growth"] = (
        yearly.groupby("Sub-Category")["Sales"]
        .pct_change()
    )

    growth = (
        yearly.groupby("Sub-Category")["Growth"]
        .mean()
        .reset_index()
    )

    growth.fillna(0,inplace=True)

    subcat = subcat.merge(
        volatility,
        on="Sub-Category"
    )

    subcat = subcat.merge(
        growth,
        on="Sub-Category"
    )

    subcat.fillna(0,inplace=True)

    X = subcat[
        [
            "Total_Sales",
            "Average_Order_Value",
            "Sales_Volatility",
            "Growth"
        ]
    ]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    subcat["Cluster"] = kmeans.fit_predict(X_scaled)

    labels = {
        0:"High Volume, Stable Demand",
        1:"Growing Demand",
        2:"Low Volume, High Volatility",
        3:"Declining Demand"
    }

    subcat["Demand Group"] = (
        subcat["Cluster"].map(labels)
    )

    st.subheader("Cluster Table")

    st.dataframe(subcat)

    pca = PCA(n_components=2)

    pca_features = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({
        "PC1":pca_features[:,0],
        "PC2":pca_features[:,1],
        "Cluster":subcat["Demand Group"],
        "Sub-Category":subcat["Sub-Category"]
    })

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_name="Sub-Category",
        size_max=15,
        title="Demand Segmentation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        "Recommendation: Maintain high inventory for "
        "'High Volume, Stable Demand' products and "
        "reduce stock for 'Declining Demand' products."
    )

    csv = subcat.to_csv(index=False).encode()

    st.download_button(
        "Download Cluster Report",
        csv,
        "clusters.csv",
        "text/csv"
    )