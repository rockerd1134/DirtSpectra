# This helper script reads test results from a file and creates a summary json object adn writes that to a summary file
# the ares are a path to a results file
# the output file will have the same filename with .summary.json appended to it

# this function takes the results and adds it the summary
def summarize_results(summary, results):
    summary["max_best_features"] = max(summary["max_best_features"], results["selector"]["best_feature_count"])
    summary["min_best_features"] = min(summary["min_best_features"], results["selector"]["best_feature_count"])
    summary["average_testing_rmse"] += results["scores"]["scaled"]["testing_rmse"]
    summary["average_unscaled_testing_rmse"] += results["scores"]["unscaled"]["testing"]
    summary["average_training_rmse"] += results["scores"]["scaled"]["training_rmse"]
    return summary

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Summarize test results')
    parser.add_argument('results_file', help='Path to results file')
    return parser.parse_args()

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

# this function takes the results and adds it the summary
def summarize_results(results):
    summary = None

    results_count = 0
    for result in results:
        results_count += 1
        if results_count == 1:
            summary = {
                "runs": 1,
                "max_best_features": result["selector"]["best_feature_count"],
                "min_best_features": result["selector"]["best_feature_count"],
                "average_testing_rmse": result["scores"]["scaled"]["testing_rmse"],
                "average_unscaled_testing_rmse": result["scores"]["unscaled"]["testing_rmse"],
                "average_training_rmse": result["scores"]["scaled"]["training_rmse"],
                # we will later add a count of which features were selected
            }
        else:
            summary["runs"] += results_count
            summary["max_best_features"] = max(summary["max_best_features"], result["selector"]["best_feature_count"])
            summary["min_best_features"] = min(summary["min_best_features"], result["selector"]["best_feature_count"])
            # the running averages
            summary["average_testing_rmse"] = (summary["average_testing_rmse"] * (results_count - 1) + result["scores"]["scaled"]["testing_rmse"]) / results_count
            summary["average_unscaled_testing_rmse"] = (summary["average_unscaled_testing_rmse"] * (results_count - 1) + result["scores"]["unscaled"]["testing_rmse"]) / results_count
            summary["average_training_rmse"] = (summary["average_training_rmse"] * (results_count - 1) + result["scores"]["scaled"]["training_rmse"]) / results_count

    return summary

if __name__ == '__main__':
    args = get_args()
    results = read_results_file(args.results_file)
    summary = summarize_results(results)
    write_results_file(summary, args.results_file + '.summary.json')