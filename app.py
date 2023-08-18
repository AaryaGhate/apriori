import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Load the dataset
data = pd.read_csv("fashion_products.csv")

# Transform data for Apriori (transaction format)
transaction_data = data.pivot_table(index='User ID', columns='Product Name', values='Rating', fill_value=0)
transaction_data = transaction_data.applymap(lambda x: 1 if x >= 4 else 0)  # Consider a purchase if rating is high

# Find frequent itemsets with Apriori
frequent_itemsets = apriori(transaction_data, min_support=0.05, use_colnames=True)

# Generate association rules
association_rules_df = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
association_rules_df = association_rules_df.sort_values(by="lift", ascending=False)

# Streamlit app
def main():
    st.set_page_config(
        page_title="Fashion Product Recommender",
        page_icon="🛍️"
    )
    
    st.title("Fashion Product Recommender with Apriori")
    st.markdown("Discover personalized fashion product recommendations using Apriori algorithm.")
    
    # User input
    user_id = st.number_input("Enter User ID", min_value=1, max_value=1000)
    
    # Get user basket (products already purchased)
    user_basket = transaction_data.loc[user_id]
    
    # Filter rules with matching antecedents
    matching_rules = association_rules_df[
        association_rules_df["antecedents"].apply(lambda x: set(x).issubset(user_basket.index))
    ]
    
    # Get recommended products
    recommended_products = matching_rules["consequents"].apply(lambda x: list(x)[0])
    
    # Display recommended products
    st.subheader("Recommended Products:")
    for product in recommended_products:
        st.write(product)

if __name__ == "__main__":
    main()
