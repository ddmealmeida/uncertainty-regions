import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from eval_pipeline import (run_subgroup_discovery, run_hierarchical_clustering, filter_redundant_subgroups,
                           plot_subgroups, run_subgroup_comparison)

run_training = False

# Read data from csv
df = pd.read_csv(r'C:\Users\daniel\.cache\kagglehub\datasets\sgpjesus\bank-account-fraud-dataset-neurips-2022\versions\2\Base.csv')
# Convert binary categorical column to numeric and select only the apropriate columns as features
df.loc[:, 'source'] = (df['source'] == 'TELEAPP').astype(int)
features = df.select_dtypes(include='number').columns.tolist()
features.remove('fraud_bool')
features.remove('month')
target = 'fraud_bool'
X = df[features].copy()
y = df[target].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=57)

# Standardize all features based on training dataset
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

if run_training:
    model_types = {
        'Logistic Regression': LogisticRegression(random_state=57, class_weight='balanced', max_iter=10000, verbose=True),
        'Random Forest': RandomForestClassifier(random_state=57, class_weight='balanced', verbose=True),
        'Gradient Boosting': GradientBoostingClassifier(random_state=57, verbose=True)
    }
    model_list = []
    errors_df = pd.DataFrame()
    for label, model in model_types.items():
        # build a binary classifier to identify fraud
        model.fit(X_train_scaled, y_train)
        model_list.append(model)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        prediction_error = abs(y_test - y_prob)
        errors_df = pd.concat([errors_df, pd.Series(prediction_error, name=label)], axis=1)
    # Plot boxplots of prediction errors for each of the three classifiers
    only_errors = errors_df.melt(var_name='class', value_name='error')
    sns.boxplot(x='class', y='error', data=only_errors)
    plt.show()
    print(only_errors.groupby('class')['error'].mean())
    errors_df.to_csv('temp_data/errors_df.csv')
else:
    errors_df = pd.read_csv('temp_data/errors_df.csv', index_col=0)

# Run the subgroup discovery for each model using Beam Search as the search algorithm
df_dict = {}
for model in errors_df.columns:
    df_dict[model] = run_subgroup_discovery(pd.DataFrame(X_test, columns=features), errors_df[model])

# Writing a LaTeX table with the subgroups found for one of the models
df_print = df_dict['Random Forest'].copy()
print(df_print.to_latex(columns=['quality', 'subgroup', 'size_sg', 'mean_sg'],
                        header=['Quality', 'Subgroup', 'Size', 'Average Error'],
                        index=False,
                        float_format="{:.3f}".format))

# Run the hierarchical clustering to measure distance between subgroups, and plot the dendrogram to visualize
ac_dict = {}
for model, df in df_dict.items():
    ac_dict[model] = run_hierarchical_clustering(df)

# Set appropriate distance thresholds
distance_thresholds = {
    'Logistic Regression': 0.25,
    'Random Forest': 0.25,
    'Gradient Boosting': 0.25
}
# Filter out the redundant subgroups
for model, df in df_dict.items():
    filtered_df = filter_redundant_subgroups(df, ac_dict[model], distance_thresholds[model])
    df_dict[model] = filtered_df.copy()

# Plot one subgroup for easier understanding
selected_subgroup = 2
# Plot the subgroups in a 2d scatterplot
plt.figure()
plot_subgroups(X_test,
               'income', 'proposed_credit_limit',
               y,
               df_dict['Random Forest'].loc[[selected_subgroup], ['subgroup', 'mean_sg', 'mean_dataset']])
plt.show()

# Finally, analyze how similar the subgroups found for each model are
run_subgroup_comparison(df_dict)