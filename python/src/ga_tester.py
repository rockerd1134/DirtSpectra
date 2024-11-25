import git
import numpy as np
import pandas as pd
import datetime
import os
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.space import Categorical, Integer, Continuous


def run_ga_test(selector_max_features=300, selector_pop_size=20, selector_generations=1):
    #for simplicty
    results = {
        "data": {
        },
        "selector": {
            "best_features": [],
            "best_feature_count": [],
        },
        "scores": {
            "scaled": {
                "training_rmse": None,
                "testing_rmse": None,
            },
            "unscaled": {
                "training_rmse": None,
                "testing_rmse": None,
            }
        }
    }
    # Testing whether using data_consol.csv helps anything. If so, probably indicates an error in reading in or joining the separate CSVs before
    repo = git.Repo('.', search_parent_directories = True)
    root = repo.working_tree_dir

    #read the data_console from a file but use os path
    data_consol = pd.read_csv(os.path.join(root, 'data', 'data_consol.csv'))
    
    X = data_consol.filter(regex="^[0-9]+$")
    bact = data_consol['pcr_bact_log']
    
    # Note: do NOT scale X and y before splitting, since that is a data leak. Instead, use the pipeline to scale both Xs, and separately scale the y for custom scoring like RMSE.
    X_train, X_test, bact_train_unscaled, bact_test_unscaled = train_test_split(X.to_numpy(), bact.to_numpy(), train_size=0.8, random_state=0)
    
    # Reshaping necessary for the y scaling step
    bact_train_unscaled = bact_train_unscaled.reshape(-1,1)
    bact_test_unscaled = bact_test_unscaled.reshape(-1,1)
    
    bact_scaler = StandardScaler()
    bact_train = bact_scaler.fit_transform(bact_train_unscaled)
    bact_test = bact_scaler.transform(bact_test_unscaled)
    
    # 10-fold CV; random state 0
    cv_10_0 = KFold(n_splits=10, shuffle=True, random_state=0)
    
    model = ElasticNet(fit_intercept=False, warm_start=True, random_state=0, selection='random', max_iter=4000)

    results["selector"]["generations"] = selector_generations
    results["selector"]["population_size"] = selector_pop_size
    
    # Define the genetic algorithm feature selector
    selector = GAFeatureSelectionCV(
        estimator=model,
        cv=10,  # Cross-validation folds
        scoring="neg_root_mean_squared_error",  # Fitness function (maximize accuracy)
        population_size=selector_pop_size,  # Number of individuals in the population
        max_features=selector_max_features,
        generations=selector_generations,  # Number of generations
        n_jobs=-1,  # Use all available CPU cores
        verbose=False,  # Print progress
    )
    
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("features", selector ),
            ("elastic_net",  model )
        ], 
        memory = root+'\\cache',
        verbose=False
    )
    
    REGULARIZATION = np.logspace(-5, 0, 8)
    MIXTURE = np.linspace(0.001, 1, 8)
    PARAM_GRID = [
        {
            "elastic_net__alpha": REGULARIZATION,
            "elastic_net__l1_ratio": MIXTURE
        }
    ]
    
    grid = GridSearchCV(estimator=pipe, param_grid=PARAM_GRID, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=cv_10_0, error_score='raise')
    grid.fit(X_train, bact_train)

    # Inverse-transforming the preds to get back to original scale.
    # Used for comparison with R results
    preds_unscaled = bact_scaler.inverse_transform(grid.predict(X_test).reshape(-1,1))

    training_rmse = round(abs(grid.score(X_train, bact_train)), 3)
    testing_rmse = round(abs(grid.score(X_test, bact_test)), 3)
    unscaled_testing_rmse = round(root_mean_squared_error(preds_unscaled, bact_test_unscaled), 3)
    print('Training RMSE:', training_rmse)
    print('Testing RMSE:', testing_rmse)
    print('Testing RMSE, unscaled:', unscaled_testing_rmse)

    results["scores"]["scaled"]["training_rmse"] = training_rmse
    results["scores"]["scaled"]["testing_rmse"] = testing_rmse
    results["scores"]["unscaled"]["testing_rmse"] = unscaled_testing_rmse

    
    
    best_pipe = grid.best_estimator_
    selector = best_pipe.named_steps['features']
    
    # Check if the selector has the 'best_features_' attribute
    if hasattr(selector, "best_features_"):
        # Get the mask of selected features (True for selected, False for not selected)
        selected_features_mask = selector.best_features_
    
        # Get the feature names (if available)
        feature_names = X_train.columns if hasattr(X_train, "columns") else [f"Feature_{i}" for i in range(X_train.shape[1])]
        selected_feature_names = [name for name, selected in zip(feature_names, selected_features_mask) if selected]

        results["selector"]["best_feature_count"] = len(selected_feature_names)
        results["selector"]["best_features"] = selected_feature_names
    
        print(f"Selected Features Count: {len(selected_feature_names)}")
        print("Selected Features:", selected_feature_names)
    else:
        print("The attribute 'best_features_' is not available.")

    return results

# this function reads a file of json results and returns the existing results object
def read_results_file(file_path='results.json'):
    import json
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

# this function overwrites the existing results file with the new results object
def write_results_file(results, file_path='results.json'):
    import json
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

# this functions reads the results, appends the new results to the existing results, and then writes
def append_results(results, file_path='results.json'):
    # fi the results file doesn't exist, create it
    if not os.path.exists(file_path):
        write_results_file([], file_path=file_path)
    existing_results = read_results_file(file_path)
    existing_results.append(results)
    write_results_file(existing_results, file_path)

# this function gaters the args using arg parse and ensure we have a total runs count, and a max_features count, and an output file location
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--max_features', type=int, default=300)
    parser.add_argument('--output_dir', type=str, default='test_run')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # set the output results_file variable to be the path for the output directory, datetime, results.json
    results_file_prefix = f'./{args.output_dir}/{int(datetime.datetime.now().timestamp())}'
    results_file = results_file_prefix + '_results.json'
    warnings_log_file = results_file_prefix + '_warnings.json'
    # create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Configure logging
    logging.basicConfig(filename=warnings_log_file, level=logging.WARNING)
    # Redirect warnings to the log file
    logging.captureWarnings(True)


    for i in range(args.runs):
        print(f"## Running Test: {i}")
        results = run_ga_test(
            selector_max_features=args.max_features,
            selector_pop_size=20,
            selector_generations=50,
        )
        append_results(results, file_path=results_file)

    print("## Done")
    print("## Results written to:", args.output_dir)