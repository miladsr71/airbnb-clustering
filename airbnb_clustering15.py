import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg') 


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


warnings.filterwarnings("ignore")

print("Loading and preparing data...")

file_path = "listings.csv"
df = pd.read_csv(file_path)

df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True).astype(float)

df = df.dropna(subset=[
    "price", 
    "number_of_reviews_ltm", 
    "number_of_reviews", 
    "calculated_host_listings_count", 
    "host_id", 
    "availability_365"
])

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR 
lower_bound = Q1 - 1.5 * IQR

df = df[
    (df['price'] > 10) & 
    (df['price'] <= upper_bound) & 
    (df['price'] >= lower_bound)
]

# Filtering out extreme host listing counts (e.g., corporate/management hosts)
df = df[df['calculated_host_listings_count'] != 594]

clustering_features = ["price", "number_of_reviews_ltm"]
X_listing = df[clustering_features].copy()

scaler = StandardScaler()
X_scaled_listing = scaler.fit_transform(X_listing)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) 
listing_clusters = kmeans.fit_predict(X_scaled_listing)
df["Listing_Cluster"] = listing_clusters

mean_prices = df.groupby("Listing_Cluster")["price"].mean().sort_values()

listing_cluster_names = {}
listing_cluster_names[mean_prices.index[0]] = "Budget"
listing_cluster_names[mean_prices.index[1]] = "Mid-range"
listing_cluster_names[mean_prices.index[2]] = "Premium"

sorted_cluster_ids = mean_prices.index.tolist()
sorted_cluster_names = [listing_cluster_names[cid] for cid in sorted_cluster_ids]


df["Listing_Name"] = df["Listing_Cluster"].map(listing_cluster_names)

cluster_summary = df.groupby("Listing_Cluster").agg(
    listing_count=('price', 'count'),
    mean_price=('price', 'mean'),
    mean_reviews_ltm=('number_of_reviews_ltm', 'mean'),
).round({'listing_count': 0, 'mean_price': 2, 'mean_reviews_ltm': 2})

cluster_summary.columns = ['count', 'price', 'reviews_ltm']

cluster_summary.index = cluster_summary.index.map(listing_cluster_names)
cluster_summary = cluster_summary.reindex(sorted_cluster_names)

print("\n--- Listing Cluster Summary (K=3) ---")
print(cluster_summary)

if "latitude" in df.columns and "longitude" in df.columns:
    custom_palette = ['red', 'blue', 'gold']
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="longitude",
        y="latitude",
        hue="Listing_Name", 
        hue_order=sorted_cluster_names, 
        palette=custom_palette, 
        data=df,
        s=20,
        alpha=0.7,
    )
    plt.title("Airbnb Listings Segmentation (K=3) - Cleaned Data")
    plt.legend(title="Listing Type") 
    plt.tight_layout()
    plt.savefig("listing_segmentation_map_cleaned.png")
    plt.close()
    print("Saved Geospatial Segmentation Map to listing_segmentation_map_cleaned.png.")
else:
    print("No latitude/longitude columns found. Skipping map plot.")
"""
plot_df = df.copy()
plot_df["Listing_Name"] = plot_df["Listing_Cluster"].map(listing_cluster_names)
category_order = ["Budget", "Mid-range", "Premium"]
plot_df["Listing_Name"] = pd.Categorical(plot_df["Listing_Name"], categories=category_order, ordered=True)

plt.figure(figsize=(8, 6))

sns.boxenplot(
    x="Listing_Name",
    y="price",
    data=plot_df,
    order=category_order,
    palette=["#FF4500", "#FFD700", "#00BFFF"] 
)

plt.title("Price Distribution per Listing Segment (Boxen Plot)")
plt.xlabel("Listing Segment")
plt.ylabel("Price (USD)")
plt.ylim(0, 500)
plt.tight_layout()
plt.savefig("price_distribution_boxplot_cleaned.png")
plt.close()
print("Saved Price Distribution Boxen Plot to price_distribution_boxplot_cleaned.png.")
"""
boxplot_data = [df[df["Listing_Cluster"] == cid]["price"] for cid in sorted_cluster_ids]

boxplot = plt.boxplot(
    boxplot_data,
    tick_labels=sorted_cluster_names, # Use descriptive names for ticks in sorted order
    patch_artist=True,
)
plt.title("Price Distribution per Cluster - Cleaned Data")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("price_distribution_boxplot_cleaned.png")
plt.close()
print("Saved Price Distribution Box Plot to price_distribution_boxplot_cleaned.png.")




print("Performing host-level clustering...")
if "host_id" in df.columns:
    df_host_prep = df.copy()

    host_df = df_host_prep.groupby("host_id").agg(
        total_listings_filtered=("price", "count"), 
        calculated_host_listings_count=("calculated_host_listings_count", "mean"),
        avg_availability=("availability_365", "mean"),
        avg_reviews_ltm=("number_of_reviews_ltm", "mean"),
    ).reset_index()

    host_features = ["calculated_host_listings_count", "avg_availability", "avg_reviews_ltm"]
    X_host = host_df[host_features]
    scaler = StandardScaler()
    X_scaled_host = scaler.fit_transform(X_host)

    kmeans_host = KMeans(n_clusters=2, random_state=42, n_init=10)
    host_clusters = kmeans_host.fit_predict(X_scaled_host)
    host_df["Host_Cluster"] = host_clusters

    host_summary = host_df.groupby("Host_Cluster").agg(
        listing_count=('total_listings_filtered', 'sum'), 
        calculated_host_listings_count=('calculated_host_listings_count', 'mean'),
        avg_availability=('avg_availability', 'mean'),
        avg_reviews_ltm=('avg_reviews_ltm', 'mean'),
    ).round(2)
    
    host_summary.columns = [
        'listing_count', 
        'mean_host_listings_count', 
        'mean_avg_availability', 
        'mean_avg_reviews_ltm'
    ]
   
    mean_listings = host_df.groupby("Host_Cluster")["total_listings_filtered"].mean().sort_values()

    host_cluster_names = {}
    host_cluster_names[mean_listings.index[0]] = "Casual"
    host_cluster_names[mean_listings.index[1]] = "Professional"
    
    sorted_host_names = [host_cluster_names[cid] for cid in mean_listings.index.tolist()]

    host_summary.index = host_summary.index.map(host_cluster_names)
    host_summary = host_summary.reindex(sorted_host_names)


    print("\n--- Host Cluster Summary (K=2) ---")
    print(host_summary)

    host_summary.to_csv("host_cluster_summary_listings_counted.csv", index=True)
    print("Saved host_cluster_summary_listings_counted.csv.")
else:
    print("No 'host_id' column found. Skipping host-level clustering.")

print("\nAll tasks completed successfully on cleaned data!")
