# A New Pipeline for Understanding Model Performance Based on Subgroup Discovery
This repository provides a tool that identifies regions with the most and least errors in Machine Learning model outputs. Using subgroup discovery techniques, it helps data scientists and ML practitioners gain insights into where their models perform well or poorly.
## Overview
The evaluation pipeline enables you to:
- Discover subgroups in your dataset where model error is significantly higher or lower than average
- Filter redundant subgroups to obtain a concise set of meaningful patterns
- Visualize subgroups in 2D plots to better understand error distributions
- Compare subgroups discovered across different models

## Installation Requirements
This project requires Python 3.9+ and the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pysubgroup
- scipy
- venn (for Venn diagrams)

You can install the dependencies using:
``` bash
pip install pandas numpy scikit-learn matplotlib seaborn pysubgroup scipy venn
```
## Usage Guide
### Quick Start
Here's how to use the evaluation pipeline with your own models:
1. Train your model(s) and generate predictions
2. Calculate prediction errors (e.g., absolute difference between predictions and ground truth)
3. Run subgroup discovery on your features and errors
4. Analyze and visualize the discovered subgroups

### Step-by-Step Example
``` python
from eval_pipeline import (run_subgroup_discovery, run_hierarchical_clustering, 
                          filter_redundant_subgroups, run_subgroup_comparison,
                          plot_subgroups_with_zoom)

# 1. Assuming you already have:
#    - X_test: your feature data as a pandas DataFrame
#    - y_test: true target values
#    - y_prob: model prediction probabilities or values
#    - prediction_error: absolute difference between predictions and ground truth

# 2. Run subgroup discovery
subgroup_results = run_subgroup_discovery(X_test, prediction_error)

# 3. Cluster similar subgroups to find patterns
clustering = run_hierarchical_clustering(subgroup_results)

# 4. Filter redundant subgroups for cleaner results
filtered_results = filter_redundant_subgroups(subgroup_results, clustering, distance_threshold=0.5)

# 5. Visualize a specific subgroup (replace with desired feature columns)
selected_subgroup = 0  # Choose the subgroup index
fig, (ax1, ax2) = plot_subgroups_with_zoom(
    X_test,
    'feature1', 'feature2',  # Replace with actual feature names
    y_test,
    filtered_results.loc[[selected_subgroup], ['subgroup', 'mean_sg', 'mean_dataset']]
)

# 6. If comparing multiple models, collect subgroup results in a dictionary and compare
model_subgroups = {
    'Model1': subgroup_results_model1,
    'Model2': subgroup_results_model2,
    # etc.
}
run_subgroup_comparison(model_subgroups)
```
## Core Functions
### `run_subgroup_discovery(features, errors, a=0.5, result_size=20, nbins=5)`
Performs subgroup discovery to identify patterns in the dataset associated with high prediction errors.
- **Parameters**:
    - `features`: DataFrame with feature values
    - `errors`: Series with prediction errors
    - `a`: Trade-off parameter between subgroup size and unusualness (default: 0.5)
    - `result_size`: Number of subgroups to return (default: 20)
    - `nbins`: Number of bins for discretizing continuous features (default: 5)

- **Returns**: DataFrame containing discovered subgroups with quality metrics

### `run_hierarchical_clustering(df)`
Performs hierarchical clustering to analyze similarity between discovered subgroups.
- **Parameters**:
    - `df`: DataFrame with subgroup information

- **Returns**: Fitted AgglomerativeClustering model

### `filter_redundant_subgroups(df, ac, distance_threshold)`
Filters redundant subgroups based on hierarchical clustering results.
- **Parameters**:
    - `df`: DataFrame with discovered subgroups
    - `ac`: Fitted AgglomerativeClustering model
    - `distance_threshold`: Maximum distance to consider subgroups as redundant

- **Returns**: Filtered DataFrame with non-redundant subgroups

### `plot_subgroups_with_zoom(data, x_column, y_column, target, subgroups)`
Creates a visualization with two plots: full view of data with subgroups highlighted, and a zoomed view of subgroup regions.
- **Parameters**:
    - `data`: DataFrame with feature data
    - `x_column`, `y_column`: Names of columns to plot on axes
    - `target`: Class values for coloring points
    - `subgroups`: DataFrame row(s) containing subgroup information

- **Returns**: Figure and axes objects

### `run_subgroup_comparison(df_dict)`
Compares subgroups discovered across different models using Venn diagrams.
- **Parameters**:
    - `df_dict`: Dictionary mapping model names to subgroup DataFrames

## Example Use Case
The repository includes a fraud detection example (see ) that demonstrates how to: `fraud_detection.py`
1. Load and preprocess a dataset
2. Train multiple model types (Logistic Regression, Random Forest, Gradient Boosting)
3. Calculate prediction errors for each model
4. Discover and analyze subgroups where errors are concentrated
5. Compare subgroups across different model types

## Advanced Usage
### Custom Quality Functions
You can modify the subgroup discovery process by changing the quality function parameter in : `run_subgroup_discovery`
``` python
task = ps.SubgroupDiscoveryTask(
    df,
    num_target,
    searchspace,
    result_set_size=result_size,
    depth=2,
    qf=ps.StandardQFNumeric(a=a))  # Try different quality functions from pysubgroup
```
### Distance Threshold Tuning
When filtering redundant subgroups, adjust the `distance_threshold` parameter to control how aggressively similar subgroups are merged:
``` python
# More aggressive filtering (fewer final subgroups)
filtered_df = filter_redundant_subgroups(df, ac, distance_threshold=0.7)

# Less aggressive filtering (more subgroups preserved)
filtered_df = filter_redundant_subgroups(df, ac, distance_threshold=0.3)
```
## Citation
If you use this pipeline in your research, please cite as instructed on the right panel of the Github Page
## License
MIT License
