import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

df = pd.read_csv("data/products.csv")
df.info()
print(df.head())

# -- PHASE 1 --
#Cleaning the data and EDA

#Cleaning the rating_stars column -> Want a float with the rate_score
#1. Extract the numerica part from the column and create the new column
df['rating_score'] = df['rating_stars'].str.extract(r'(\d+\.?\d*)')

#2. Convert the new column to float and deal with the NaN
df['rating_score'] = pd.to_numeric(df['rating_score'])

#3. Verification
print(df[['rating_stars', 'rating_score']].head(10))

#Deal with Nan
df.dropna(subset=["rating_score"], inplace=True)


#Estimates of location
mean_rating_score = df["rating_score"].mean()
print(f"The mean of the ratio score is {mean_rating_score:.2f}")
median_rating_score = df["rating_score"].median()
print(f"The median of the ratio score is {median_rating_score}")
"""
#Histogram
ax = df["rating_score"].plot.hist(figsize=(5, 5), bins=30)
ax.set_xlabel("Rating Score")
plt.savefig("visualizations/rating_score_histogram.png")
plt.show()

#Analysis X columns
#price_value
#Histogram
ax = df["price_value"].plot.hist(figsize=(5,5 ), bins=30)
ax.set_xlabel("Price Value")
plt.savefig("visualizations/price_value_histogram")
plt.show()
"""
#Estimates of location
mean_price_value = df["price_value"].mean()
print(f"The mean of the price value is {mean_price_value:.2f}")
median_price_value = df["price_value"].median()
print(f"The median of the price value is {mean_price_value}")

#rank_1
#Histogram
ax = df["rank_1"].plot.hist(figsize=(5,5), bins=30)
ax.set_xlabel("Rank")
plt.show()

# --- Imputing 'rank_1' ---
# Calculate the median rank
median_rank = df['rank_1'].median()
print(f"Median rank: {median_rank}")


# Fill NaNs with the median
df['rank_1'] = df['rank_1'].fillna(median_rank)

#rating_count
#Cleaning the column
df['rating_count_int'] = df['rating_count'].str.replace(' rating', '') \
    .str.replace('s', '') \
    .str.replace(',', '')

# Now, convert to numeric, fill NaNs, and convert to integer
df['rating_count_int'] = pd.to_numeric(df['rating_count_int'])

df['rating_count_int'] = df['rating_count_int'].fillna(0)

df['rating_count_int'] = df['rating_count_int'].astype(int)

#Verification
print(df["rating_count_int"].head())

#Histogram
ax = df["rating_count_int"].plot.hist(figsize=(5, 5), bins=50)
ax.set_xlabel("Rating Count")
plt.savefig("visualizations/rating_count_histogram.png")
plt.show()

mean_rating_count = df["rating_count_int"].mean()
print(f"The mean of the rating count is {mean_rating_count:.2f}")
median_rating_count = df["rating_count_int"].median()
print(f"The median of the rating count is {median_rating_count:.2f}")

#brand_name
#Top 20 most frequent brands
top_20_brands = df['brand_name'].value_counts().head(20).index
print(f"Top 20 most frequent brands:\n{top_20_brands.to_list()}")

# 2. Filter the DataFrame to ONLY include these top 20 brands
top_20_df = df[df['brand_name'].isin(top_20_brands)]

# 3. Calculate the average rating AND count for each of these top brands
# We sort by 'rating_score' to see who is best
brand_analysis = top_20_df.groupby('brand_name')['rating_score'].agg(['mean', 'count']).sort_values(by='mean',
                                                                                                    ascending=False)
print("\n--- Analysis of Top 20 Brands (Sorted by Average Rating) ---")
print(brand_analysis.to_markdown(floatfmt=".2f"))

# 2. Create a new column 'brand_name_encoded'
# Use .apply() with a lambda function:
# - IF the brand is in our top 20 list, keep its name.
# - ELSE, replace it with 'Other'.
df['brand_name_encoded'] = df['brand_name'].apply(lambda x: x if x in top_20_brands else 'Other')

# 3. Show the result
print("\n--- Feature Transformation Result ---")
print(df[['brand_name', 'brand_name_encoded']].head(10).to_markdown(index=False))

print(f"\nOriginal unique brands: {df['brand_name'].nunique()}")
print(f"New unique categories:  {df['brand_name_encoded'].nunique()}")

#Box Plot
ax = df.boxplot(by="brand_name_encoded", column="rating_score")
ax.set_xlabel("Brand Name")
ax.set_ylabel("Rating Score")
plt.show()

#Seller name
#Top 20 most frequent sellers
top_20_sellers = df['seller_name'].value_counts().head(20).index
print(f"Top 20 most frequent sellers:\n{top_20_sellers.to_list()}")

# 2. Filter the DataFrame to ONLY include these top 20 sellers
top_20_df = df[df['seller_name'].isin(top_20_sellers)]

# 3. Calculate the average rating AND count for each of these top sellers
# We sort by 'rating_score' to see who is best
seller_analysis = top_20_df.groupby('seller_name')['rating_score'].agg(['mean', 'count']).sort_values(by='mean',
                                                                                                    ascending=False)
print("\n--- Analysis of Top 20 Sellers (Sorted by Average Rating) ---")
print(seller_analysis.to_markdown(floatfmt=".2f"))

# 2. Create a new column 'seller_name_encoded'
# Use .apply() with a lambda function:
# - IF the seller is in our top 20 list, keep its name.
# - ELSE, replace it with 'Other'.
df['seller_name_encoded'] = df['seller_name'].apply(lambda x: x if x in top_20_sellers else 'Other')

# 3. Show the result
print("\n--- Feature Transformation Result ---")
print(df[['seller_name', 'seller_name_encoded']].head(10).to_markdown(index=False))

print(f"\nOriginal unique sellers: {df['seller_name'].nunique()}")
print(f"New unique categories:  {df['seller_name_encoded'].nunique()}")

#breadcrumbs
print("--- Part 1: Filtering ---")

# Step 1.1: Fill missing breadcrumbs BEFORE splitting
df['breadcrumbs'] = df['breadcrumbs'].fillna('Unknown')

# Step 1.2: Define the robust extraction function
def extract_breadcrumb_level(breadcrumb_str, level=0):
    """
    Safely extracts a specific level from the breadcrumb string.
    """
    try:
        parts = breadcrumb_str.split(' › ')
        if len(parts) > level:
            return parts[level].strip()
        else:
            return 'None' # Return 'None' if the level doesn't exist
    except:
        return 'None'

# Step 1.3: Extract Level 0 (the top category)
df['L0_Category'] = df['breadcrumbs'].apply(lambda x: extract_breadcrumb_level(x, level=0))

print(f"Original shape of data: {df.shape}")
print("Top-level categories before filtering:")
print(df['L0_Category'].value_counts())

# Step 1.4: Filter the DataFrame and create a clean .copy()
df_clothing = df[df['L0_Category'] == 'Clothing, Shoes & Jewelry'].copy()

print(f"\nNew shape of (df_clothing): {df_clothing.shape}")


# --- PART 2: EXTRACTING THE HIERARCHY ---

print("\n\n--- Part 2: Extracting Hierarchy ---")

# We now work ONLY on the df_clothing DataFrame

# Step 2.1: Apply the function for Level 1 ("Audience": Men, Women, etc.)
df_clothing['L1_Audience'] = df_clothing['breadcrumbs'].apply(lambda x: extract_breadcrumb_level(x, level=1))

# Step 2.2: Apply the function for Level 2 ("Product Type": Clothing, Shoes, etc.)
df_clothing['L2_ProductType'] = df_clothing['breadcrumbs'].apply(lambda x: extract_breadcrumb_level(x, level=2))

# Step 2.3: Apply the function for Level 3 ("Subtype": Active, Shirts, etc.)
df_clothing['L3_Subtype'] = df_clothing['breadcrumbs'].apply(lambda x: extract_breadcrumb_level(x, level=3))


# --- VERIFICATION ---
print("\n--- Verification of New Features ---")

# Select a sample of columns to show the result
print(df_clothing[['breadcrumbs', 'L1_Audience', 'L2_ProductType', 'L3_Subtype']].head(10).to_markdown(index=False))

# Show the new granular categories we've created
print("\nNew 'L1_Audience' categories:")
print(df_clothing['L1_Audience'].value_counts().to_markdown())

print("\nNew 'L2_ProductType' categories:")
print(df_clothing['L2_ProductType'].value_counts().to_markdown())

print("\nNew 'L3_Subtype' categories:")
print(df_clothing['L3_Subtype'].value_counts().head(10).to_markdown())
print(df_clothing["L3_Subtype"].head(10))

#Link the column with the original dataframe
df["L3_Subtype"] = df_clothing["L3_Subtype"]

#Text variables
#Deal with Nan
df["customer_review_summary"] = df["customer_review_summary"].fillna("")

#Calculate the length of the entries
# Create 'title_length'
df['title_length'] = df['title'].str.len()

# Create 'about_item_length'
df['about_item_length'] = df['about_item'].str.len()

# Create 'review_summary_length'
df['customer_review_summary_length'] = df['customer_review_summary'].str.len()

# --- Analysis ---
# Now, see if there is any correlation with our target 'rating_score'
# A positive correlation means longer text = higher rating
# A negative correlation means longer text = lower rating
print(df[['title_length', 'about_item_length', 'customer_review_summary_length', 'rating_score']].corr())

# Create the 'super-text' column
df['text_features'] = df['title'] + ' ' + df['about_item'] + ' ' + df['customer_review_summary']

# --- Add the bullet point feature ---
# Count the number of newline characters '\n' and add 1
df['about_item_bullets'] = df['about_item'].str.count('.') + 1

# Correct for empty strings (which would be 1 but should be 0)
df.loc[df['about_item_length'] == 0, 'about_item_bullets'] = 0

# --- Dropping Noisy/Empty Columns ---
# We also drop the original text columns since we engineered new features from them
original_text_cols = ['title', 'about_item', 'customer_review_summary', 'breadcrumbs', 'rating_stars']
cols_to_drop = ['list_price', 'product_description', "best_sellers_rank"] + original_text_cols

# We use errors='ignore' in case a column was already dropped
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

print(f"--- DataFrame after dropping columns ---")
df.info()


#-- BUILD THE MODEL --

# --- Step 1: Define your feature (X) and target (y) ---

# Select the target variable (Y)
y = df['rating_score']

# Select the predictor variables (X)
# We select ONLY the final, modified columns
numeric_features = [
    'price_value',
    'rank_1',
    'rating_count_int',
    'title_length',
    'about_item_length',
    'customer_review_summary_length',
    'about_item_bullets'
]

categorical_features = [
    'brand_name_encoded',
    'seller_name_encoded',
    'L3_Subtype'
]

# This is the 'super-text' column we created
text_feature = 'text_features'

# Join all the predictors
X = df[numeric_features + categorical_features + [text_feature]]

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --- Step 3: Build the Preprocessing Pipelines ---

# 1. Numeric Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 2. Categorical Pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 3. Text Pipeline
text_transformer = TfidfVectorizer(stop_words='english', max_features=200)

# --- Step 4: Create the ColumnTransformer ---

# This object takes our 3 pipelines and applies them to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, text_feature)
    ],
    remainder='passthrough'
)

# --- Step 5: Build Final Model Pipelines ---

# --- Model 1: Linear Regression ---
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# --- Model 2: Random Forest ---
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

#Models' ejecution
# --- Step 1: Train the Models ---
print("Starting model training...")

# Train the Linear Regression pipeline
lr_pipeline.fit(X_train, y_train)
print("Linear Regression model trained.")

# Train the Random Forest pipeline
rf_pipeline.fit(X_train, y_train)
print("Random Forest model trained.")

# --- Step 2: Evaluate Linear Regression ---
print("\n--- Linear Regression Performance ---")
# Make predictions on the test set
lr_preds = lr_pipeline.predict(X_test)

# Calculate metrics
lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

print(f"R-squared (R²): {lr_r2:.4f}")
print(f"Mean Absolute Error (MAE): {lr_mae:.4f} stars")
print(f"Root Mean Squared Error (RMSE): {lr_rmse:.4f} stars")

# --- Step 3: Evaluate Random Forest ---
print("\n--- Random Forest Performance ---")
# Make predictions on the test set
rf_preds = rf_pipeline.predict(X_test)

# Calculate metrics
rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print(f"R-squared (R²): {rf_r2:.4f}")
print(f"Mean Absolute Error (MAE): {rf_mae:.4f} stars")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.4f} stars")

# --- Step 4: Get Feature Names for Interpretation ---

# Get the 'preprocessor' from one of the pipelines
preprocessor = lr_pipeline.named_steps['preprocessor']

# Feature names from the OneHotEncoder
one_hot_features = preprocessor.named_transformers_['cat'] \
                               .named_steps['encoder'] \
                               .get_feature_names_out(categorical_features)

# Feature names from the TfidfVectorizer
text_features = preprocessor.named_transformers_['text'].get_feature_names_out()

# Combine all feature names in the correct order
all_feature_names = numeric_features + list(one_hot_features) + list(text_features)

print(f"\nTotal features created by preprocessor: {len(all_feature_names)}")

# Get the trained regressor
lr_model = lr_pipeline.named_steps['regressor']

# Combine feature names with their coefficients
lr_coeffs = pd.Series(lr_model.coef_, index=all_feature_names)

print("\n--- Linear Regression Model 'Summary' (Coefficients) ---")
print(f"Intercept (Base Rating): {lr_model.intercept_:.4f}\n")

print("Top 10 Positive Features (Increase Rating):")
print(lr_coeffs.nlargest(10).to_markdown(floatfmt=".4f"))

print("\nTop 10 Negative Features (Decrease Rating):")
print(lr_coeffs.nsmallest(10).to_markdown(floatfmt=".4f"))

# Get the trained regressor
rf_model = rf_pipeline.named_steps['regressor']

# Combine feature names with their importances
rf_importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)

print("\n--- Random Forest Model 'Summary' (Top 20 Features) ---")
print(rf_importances.nlargest(20).to_markdown(floatfmt=".4f"))