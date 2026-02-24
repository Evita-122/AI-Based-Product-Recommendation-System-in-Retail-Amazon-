import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Amazon AI Recommender", layout="wide")

# ---------------------------------
# EXTENDED PRODUCT CATALOG
# ---------------------------------
products = {
    "Laptop": {"price": "₹60,000"},
    "Gaming Laptop": {"price": "₹95,000"},
    "Mouse": {"price": "₹500"},
    "Wireless Mouse": {"price": "₹900"},
    "Keyboard": {"price": "₹1,200"},
    "Mechanical Keyboard": {"price": "₹3,500"},
    "Headphones": {"price": "₹2,500"},
    "Bluetooth Headphones": {"price": "₹4,000"},
    "Smartphone": {"price": "₹25,000"},
    "Tablet": {"price": "₹18,000"},
    "Smartwatch": {"price": "₹7,000"},
    "Monitor": {"price": "₹12,000"},
    "Printer": {"price": "₹8,500"},
    "Webcam": {"price": "₹2,200"},
    "External Hard Drive": {"price": "₹5,500"},
}

# ---------------------------------
# EXTENDED USER RATING DATA
# ---------------------------------
data = {
    'user_id': [
        1,1,1,1,
        2,2,2,2,
        3,3,3,3,
        4,4,4,4,
        5,5,5,5,
        6,6,6,6,
        7,7,7,
        8,8,8,
        9,9,9,
        10,10,10
    ],
    'product_id': [
        "Laptop","Mouse","Keyboard","Monitor",
        "Gaming Laptop","Mechanical Keyboard","Mouse","Headphones",
        "Smartphone","Bluetooth Headphones","Smartwatch","Tablet",
        "Laptop","Wireless Mouse","Printer","Webcam",
        "Tablet","Smartphone","Smartwatch","Headphones",
        "Monitor","External Hard Drive","Printer","Webcam",
        "Gaming Laptop","Mechanical Keyboard","Monitor",
        "Smartphone","Tablet","Bluetooth Headphones",
        "Laptop","External Hard Drive","Keyboard",
        "Smartwatch","Webcam","Wireless Mouse"
    ],
    'rating': [
        5,4,4,5,
        5,4,4,5,
        5,5,4,4,
        4,5,4,4,
        5,4,5,4,
        4,5,4,4,
        5,4,5,
        4,5,4,
        5,4,4,
        4,5,4
    ]
}

df = pd.DataFrame(data)

# ---------------------------------
# CREATE USER-ITEM MATRIX
# ---------------------------------
user_item_matrix = df.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating'
).fillna(0)

# ---------------------------------
# SIMILARITY
# ---------------------------------
similarity = cosine_similarity(user_item_matrix)

similarity_df = pd.DataFrame(
    similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ---------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------
def recommend_products(user_id, top_n=6):

    similar_users = similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)

    recommendations = {}

    for similar_user in similar_users.index:
        products_rated = user_item_matrix.loc[similar_user]

        for product, rating in products_rated.items():
            if user_item_matrix.loc[user_id, product] == 0 and rating >= 4:
                recommendations[product] = rating

    recommended_products = sorted(
        recommendations,
        key=recommendations.get,
        reverse=True
    )

    return recommended_products[:top_n]


# ---------------------------------
# UI
# ---------------------------------
st.title("🛒 Amazon AI Recommendation System (Extended Version)")

user_id = st.selectbox("Select User", user_item_matrix.index)

if st.button("Generate Recommendations"):

    results = recommend_products(user_id)

    st.subheader("🎯 Recommended For You")

    if results:
        cols = st.columns(3)
        for idx, product in enumerate(results):
            with cols[idx % 3]:
                st.markdown(f"""
                <div style='
                    background:#1f2937;
                    padding:20px;
                    border-radius:15px;
                    color:white;
                    margin-bottom:15px;
                '>
                    <h4>{product}</h4>
                    <p style='color:#f59e0b;font-weight:bold;'>
                        {products[product]["price"]}
                    </p>
                    ⭐⭐⭐⭐☆
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No recommendations available.")