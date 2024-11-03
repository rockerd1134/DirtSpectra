# RKNGH project for root knot nematode greenhouse stress, translated from R to Python.
# This code is adapted for Google Colab or AWS SageMaker environments, using Python libraries.

# -----
# SETUP
# -----

# Import necessary libraries for data handling, visualization, and machine learning.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import joblib  # For caching models
import os
import seaborn as sns
import warnings
import re
warnings.filterwarnings("ignore")

# Set seed for reproducibility
SEED = 0
np.random.seed(SEED)

# -----
# DATA IMPORTING AND CLEANING
# -----

def merge_datasets(df1, df2, df1_id=None, df2_id="Sample", new_id="sample"):
    """
    Merges two datasets on a specified identifier column and cleans up redundant columns.
    If df1_id is None, the first unnamed column in df1 will be used by default.

    Parameters:
    - df1: The first DataFrame to merge (e.g., `stress_pca_psr`).
    - df2: The second DataFrame to merge (e.g., `stress`).
    - df1_id: The column in `df1` to use for merging. If None, defaults to the first unnamed column.
    - df2_id: The column in `df2` to use for merging (default is 'Sample').
    - new_id: The name for the identifier column in the merged DataFrame (default is 'sample').

    Returns:
    - A merged DataFrame with a single identifier column.
    """
    
    # If df1_id is not provided, use the first unnamed column in df1
    if df1_id is None:
        # Find the first unnamed column
        unnamed_cols = [col for col in df1.columns if "Unnamed" in col]
        if unnamed_cols:
            df1_id = unnamed_cols[0]
        else:
            raise ValueError("No unnamed columns found in df1; please specify df1_id explicitly.")

    # Rename the specified identifier column in df1 to `new_id`.
    df1 = df1.rename(columns={df1_id: new_id})

    # Ensure the identifier column in df1 is of integer type to match df2 for merging.
    df1[new_id] = df1[new_id].astype(int)

    # Perform an inner join on the specified columns, retaining only matching rows.
    merged_data = df1.merge(df2, left_on=new_id, right_on=df2_id, how="inner")

    # Drop the redundant identifier column from df2 to avoid duplication.
    merged_data = merged_data.drop(columns=df2_id)

    return merged_data

def preprocess_data(data):
    # Remove columns that start with "Mean"
    data = data.loc[:, ~data.columns.str.startswith("Mean")]
    
    # Add the 'obs' column as a repeating sequence from 1 to 5 (80 times)
    data["obs"] = np.tile(range(1, 6), len(data) // 5)
    
    # Reorder the columns to place specified columns at the beginning
    cols_to_relocate = ["sample", "obs" ,"TRT","Block","Irrigation","Inoculation","Resistance"]
    data = data[cols_to_relocate + [col for col in data.columns if col not in cols_to_relocate]]
    
    # Filter out specific outliers identified in the EDA phase
    data = data[~((data["sample"] == 55) & (data["obs"] == 1)) & ~((data["sample"] == 71) & (data["obs"] == 1))]
    
    # Reorder the levels of the 'irrig' column, if it's categorical, so "Half" comes before other levels
    if data["Irrigation"].dtype.name == "category":
        data["Irrigation"] = data["Irrigation"].cat.reorder_categories(["Half"] + [lvl for lvl in data["Irrigation"].cat.categories if lvl != "Half"])
    
    # Add an 'id' column as a unique identifier for each row
    data = data.reset_index(drop=True)
    data["id"] = data.index + 1  # +1 to start 'id' from 1

    # 1. Create data subsets for classification and regression tasks
    
    # Classification subset: Select relevant columns for classification task
    hs_irrig = data[["id", "sample", "obs", "Irrigation"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    
    # Regression subsets: Select relevant columns for each regression target variable
    hs_bact = data[["id", "sample", "obs", "Log16S"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    hs_cbblr = data[["id", "sample", "obs", "Logcbblr"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    hs_fungi = data[["id", "sample", "obs", "Log18S"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    hs_phoa = data[["id", "sample", "obs", "Logphoa"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    hs_urec = data[["id", "sample", "obs", "Ratiourec"] + [col for col in data.columns if re.match(r"^[0-9]+$", col)]].dropna()
    
    # 2. Split each subset into training and test sets with an 80/20 split
    
    # Classification split
    X_irrig = hs_irrig.drop(columns="Irrigation")
    y_irrig = hs_irrig["Irrigation"]
    X_train_irrig, X_test_irrig, y_train_irrig, y_test_irrig = train_test_split(X_irrig, y_irrig, test_size=0.2, random_state=SEED)
    
    # Regression splits
    X_bact = hs_bact.drop(columns="Log16S")
    y_bact = hs_bact["Log16S"]
    X_train_bact, X_test_bact, y_train_bact, y_test_bact = train_test_split(X_bact, y_bact, test_size=0.2, random_state=SEED)
    
    X_cbblr = hs_cbblr.drop(columns="Ratiocbblr")
    y_cbblr = hs_cbblr["Ratiocbblr"]
    X_train_cbblr, X_test_cbblr, y_train_cbblr, y_test_cbblr = train_test_split(X_cbblr, y_cbblr, test_size=0.2, random_state=SEED)
    
    X_fungi = hs_fungi.drop(columns="Ratiofungi")
    y_fungi = hs_fungi["Ratiofungi"]
    X_train_fungi, X_test_fungi, y_train_fungi, y_test_fungi = train_test_split(X_fungi, y_fungi, test_size=0.2, random_state=SEED)
    
    X_phoa = hs_phoa.drop(columns="Logphoa")
    y_phoa = hs_phoa["Logphoa"]
    X_train_phoa, X_test_phoa, y_train_phoa, y_test_phoa = train_test_split(X_phoa, y_phoa, test_size=0.2, random_state=SEED)
    
    X_urec = hs_urec.drop(columns="Logurec")
    y_urec = hs_urec["Logurec"]
    X_train_urec, X_test_urec, y_train_urec, y_test_urec = train_test_split(X_urec, y_urec, test_size=0.2, random_state=SEED)

    return data




# Load data (replace with appropriate path on your environment)
data_dir = "./data"
stress = pd.read_csv(f"{data_dir}/RKNGHStress.csv")
stress_pca_psr = pd.read_csv(f"{data_dir}/RKNGHStressPCAPSR.csv")

data = merge_datasets(stress_pca_psr, stress)

# commented this out because the columns are already renamed in our current data
# Clean and rename columns for consistency and readability.
#data.rename(columns={
#    "TRT": "trt_code", "Block": "block", "Irrigation": "irrig", "Inoculation": "inoc",
#    # ... additional renaming following your R code convention ...
#}, inplace=True)

data = preprocess_data(data)

# -----
# HELPER FUNCTIONS
# -----

# Cache function to store/load models and results
def cache(funct, fname):
    if not fname.startswith("cache/"):
        fname = "cache/" + fname
    if os.path.exists(fname):
        return joblib.load(fname)
    else:
        result = funct()
        joblib.dump(result, fname)
        return result

# Function to plot ROC curves
def plot_roc_curve(model, X_test, y_test, model_name):
    y_probs = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC Curve")
    plt.legend()

# Function for tuning and fitting models (generalized for different model types)
def tune_fit(model_type, mode, scoring, data_split, formula, corr_thresh=None, verbose=False):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data_split.drop(columns=[formula]), data_split[formula], test_size=0.2, random_state=SEED
    )

    # Initialize the model and grid based on the type specified
    if model_type == "lasso":
        model = Lasso()
        param_grid = {"alpha": np.logspace(-4, 0, 10)}
    elif model_type == "elast":
        model = ElasticNet()
        param_grid = {"alpha": np.logspace(-4, 0, 10), "l1_ratio": [0.1, 0.5, 0.9]}
#    elif model_type == "Random forest":
#        if mode == "classification":
#            model = RandomForestClassifier()
#            param_grid = {"n_estimators": [100, 200], "max_features": ["sqrt", "log2"]}
#        else:
#            model = RandomForestRegressor()
#            param_grid = {"n_estimators": [100, 200], "max_features": ["sqrt", "log2"]}
    else:
        raise ValueError("Unknown model type")

    # Grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=5, verbose=verbose)
    grid_search.fit(X_train, y_train)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    scores = cross_val_score(best_model, X_test, y_test, scoring=scoring)

    if verbose:
        print(f"Best parameters for {model_type}: {best_params}")
        print(f"Cross-validation scores: {scores}")
    return best_model, best_params

# -----
# MODEL TRAINING AND EVALUATION
# -----

# Example usage for logistic regression and random forest (classification)
X = data.drop(columns="irrig")  # Features
y = data["irrig"]               # Target variable

# Logistic Regression (Lasso)
lasso_model, lasso_params = tune_fit("lasso", "classification", make_scorer(f1_score), X, y, verbose=True)

# Random Forest (Classification)
#rf_model, rf_params = tune_fit("Random forest", "classification", make_scorer(roc_auc_score), X, y, verbose=True)


# Plotting ROC Curve for Classification Models
plt.figure(figsize=(10, 6))
plot_roc_curve(lasso_model, X_test, y_test, "Lasso Logistic Regression")
plot_roc_curve(rf_model, X_test, y_test, "Random Forest")
plt.show()
