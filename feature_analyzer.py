# Import pandas for data manipulation
import pandas as pd
# Import numpy for numerical operations
import numpy as np
# Import pyplot from matplotlib for plotting
import matplotlib.pyplot as plt
# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import PCA for dimensionality reduction
from sklearn.decomposition import PCA
# Import LinearDiscriminantAnalysis for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Import RFE for recursive feature elimination
from sklearn.feature_selection import RFE
# Import LogisticRegression for RFE estimator
from sklearn.linear_model import LogisticRegression

# Define function for feature analysis
def analyze_features(df):
    # Print start label
    print("\n==========================================")
    # Print start label
    print("O/P: Starting Phase 2 (Feature Analysis)...")
    
    # --- Week 1 style — basic data handling ---
    # Print the output label
    print("O/P:")
    # Print the dimensions of the DataFrame
    print(df.shape)
    # Print the first 5 rows of the DataFrame
    print(df.head())
    # Print the column names of the DataFrame
    print(df.columns)
    # Print the count of missing values per column
    print(df.isnull().sum())

    # --- Week 2 style — descriptive statistics ---
    # Print the output label
    print("O/P:")
    # Calculate the 25th percentile (Q1) of token_length
    Q1 = df['token_length'].quantile(0.25)
    # Calculate the 75th percentile (Q3) of token_length
    Q3 = df['token_length'].quantile(0.75)
    # Calculate the Interquartile Range (IQR)
    IQR = Q3 - Q1
    # Print the mean of token_length
    print("Mean:", df['token_length'].mean())
    # Print the standard deviation of token_length
    print("Std Dev:", df['token_length'].std())
    # Print the variance of token_length
    print("Variance:", df['token_length'].var())
    # Print the calculated IQR
    print("IQR:", IQR)

    # --- Week 3 style — plots ---
    # Print the output label
    print("O/P: Plots generated and saved to disk.")
    # Create a new matplotlib figure
    plt.figure()
    # Plot a box plot for token_length, dropping missing values
    plt.boxplot(df['token_length'].dropna())
    # Set the title for the box plot
    plt.title("Box plot of Token Length")
    # Save the plot
    plt.savefig("plot_box.png")

    # Create another matplotlib figure
    plt.figure()
    # Plot a histogram for specificity_score with 20 bins
    plt.hist(df['specificity_score'].dropna(), bins=20)
    # Set the title for the histogram
    plt.title("Histogram of Specificity Score")
    # Save the plot
    plt.savefig("plot_hist.png")

    # Create another matplotlib figure
    plt.figure()
    # Plot a scatter plot of token_length versus instruction_count
    plt.scatter(df['token_length'], df['instruction_count'])
    # Set the title for the scatter plot
    plt.title("Token Length vs Instruction Count")
    # Set the X-axis label
    plt.xlabel("Token Length")
    # Set the Y-axis label
    plt.ylabel("Instruction Count")
    # Save the plot
    plt.savefig("plot_scatter.png")

    # --- Week 4 style — data preprocessing ---
    # Print the output label
    print("O/P:")
    # Fill missing values in specificity_score with its mean
    df['specificity_score'] = df['specificity_score'].fillna(df['specificity_score'].mean())
    # Calculate the lower bound for outlier detection
    lower = Q1 - 1.5 * IQR
    # Calculate the upper bound for outlier detection
    upper = Q3 + 1.5 * IQR
    # Filter out token_length outliers based on IQR bounds
    df = df[(df['token_length'] >= lower) & (df['token_length'] <= upper)].copy()
    # Cap extreme token_length values at lower and upper bounds
    df['token_length'] = df['token_length'].clip(lower=lower, upper=upper)

    # --- Week 5 style — feature construction ---
    # Print the output label
    print("O/P:")
    # Feature Extraction (Instructions & Specificity)
    instruction_verbs = ['always','never','do','don\'t','start','follow','include','end','keep','ensure','must','strictly','exactly']
    specificity_keywords = ["must", "only", "strictly", "require", "exactly", "specific", "detailed", "ensure", "always", "never"]

    def count_instructions(text):
        count = 0
        for line in str(text).lower().split('\n'):
            line = line.strip()
            if any(line.startswith(str(i)) for i in range(10)) or any(line.startswith(v) for v in instruction_verbs):
                count += 1
        return count

    df['instruction_count'] = df['prompt_text'].apply(count_instructions)
    df['specificity_score'] = df['prompt_text'].apply(lambda x: (sum(1 for word in str(x).lower().split() if word in specificity_keywords) / len(str(x).split()) * 100) if len(str(x).split()) > 0 else 0)
    # Create a new column 'prompt_type' based on instruction_count
    df['prompt_type'] = np.where(df['instruction_count'] > 5, "detailed", "simple")
    # Generate dummy variables for the 'prompt_type' column
    dummies = pd.get_dummies(df['prompt_type'], prefix='type')
    # Concatenate the dummy variables back into the DataFrame
    df = pd.concat([df, dummies], axis=1)
    # Create an ordinally encoded version of 'prompt_type'
    df['prompt_type_encoded'] = df['prompt_type'].map({'simple': 1, 'detailed': 2})
    # Bin 'token_length' into 'Short', 'Medium', 'Long' categories
    df['token_length_binned'] = pd.cut(df['token_length'], bins=[0, 50, 150, 500, np.inf], labels=['Short', 'Medium', 'Long', 'ExtraLong'])

    # --- Week 6 style — feature extraction ---
    # Print the output label
    print("O/P:")
    # Define the list of numeric columns to extract
    numeric_cols = ['token_length', 'instruction_count', 'specificity_score', 'example_count', 'constraint_density']
    # Subset the DataFrame to get only numeric features
    X = df[numeric_cols]
    # Set the target variable for LDA and RFE
    y = df['prompt_type_encoded']

    # Initialize a StandardScaler object
    scaler = StandardScaler()
    # Fit and transform the numeric features
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA with 2 components
    pca = PCA(n_components=2)
    # Fit and transform the scaled data using PCA
    X_pca = pca.fit_transform(X_scaled)
    # Print the explained variance ratio of the PCA components
    print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Perform Singular Value Decomposition (SVD) on the scaled data
    U, S, VT = np.linalg.svd(X_scaled)
    # Print the singular values from SVD
    print("SVD Singular Values:", S)

    # Check if there is more than one unique class in target variable
    if len(y.dropna().unique()) > 1:
        # Initialize LinearDiscriminantAnalysis with 1 component
        lda = LinearDiscriminantAnalysis(n_components=1)
        # Fit and transform the data using LDA
        X_lda = lda.fit_transform(X_scaled, y)
        # Print success message for LDA
        print("LDA executed successfully.")
    else:
        # Print skip message if not enough classes
        print("Skipping LDA: only one class present.")

    # Check if there is more than one unique class in target variable
    if len(y.dropna().unique()) > 1:
        # Initialize a LogisticRegression estimator with max 200 iterations
        estimator = LogisticRegression(max_iter=200)
        # Initialize Recursive Feature Elimination (RFE) to select 2 features
        rfe = RFE(estimator, n_features_to_select=2)
        # Fit the RFE model
        rfe.fit(X_scaled, y)
        # Print the feature ranking from RFE
        print("RFE Feature Ranking:", rfe.ranking_)
    else:
        # Print skip message if not enough classes
        print("Skipping RFE: only one class present.")
        
    # Return the processed DataFrame
    return df
