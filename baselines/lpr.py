# File based on the code sent by Patrícia and Ricardo from UFPE, used in the Local Performance Regions paper
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Iterable
from scipy.stats import beta
from sklearn.tree import DecisionTreeRegressor


# Funções que extraem as regras geradas pela Árvore de Decisão
def find_path(clf,node_numb, path, x):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    path.append(node_numb)
    if node_numb == x:
        return True
    left = False
    right = False
    if (children_left[node_numb] !=-1):
        left = find_path(clf,children_left[node_numb], path, x)
    if (children_right[node_numb] !=-1):
        right = find_path(clf, children_right[node_numb], path, x)
    if left or right :
        return True
    path.remove(node_numb)
    return False


def get_rule(clf, path, column_names):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    mask = ''
    for index, node in enumerate(path):
        #We check if we are not in the leaf
        if index!=len(path)-1:
        # Do we go under or over the threshold ?

            if (children_left[node] == path[index+1]):
                mask += "(data['{}'] <= {}) \t ".format(column_names[feature[node]], threshold[node])
            else:
                mask += "(data['{}'] > {}) \t ".format(column_names[feature[node]], threshold[node])
    # We insert the & at the right places
    mask = mask.replace("\t", "&", mask.count("\t") - 1)
    mask = mask.replace("\t", "")
    return mask


def rules(clf, X_t):
    column_names = X_t.columns
    leave_id = clf.apply(X_t)

    paths ={}
    k = []
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(clf,0, path_leaf, leaf)
        paths[leaf] = np.unique(np.sort(path_leaf))
        k.append(clf.tree_.value[leaf])

    rules = []
    for key in paths:
        rules.append(get_rule(clf, paths[key], column_names))
    regras = pd.DataFrame(columns = ['antecedent', 'consequent','value'])
    for n in range(len(k)):
        regras.loc[len(regras),:] = [rules[n], '(data[\'erro_pred\'] == ' + str(k[n][0][0])[:4] + ')',k[n][0][0]]

    return regras

def get_max_min(regras, data):

    df_part  = pd.DataFrame(columns = ['antecedent', 'consequent', 'minimum','maximum','confidence'])
    for n in range(len(regras)):
        if regras.iloc[n,0] == '':
            continue
        df = data[eval(regras.iloc[n,0])]
        df = df.reset_index(drop = True)

        try:
            a, b, loc, scale = beta.fit(df.errors)
            minimum_value = beta.interval(0.95, a, b, loc=0, scale=1)[0]
            maximum_value = beta.interval(0.95, a, b, loc=0, scale=1)[1]
        except:
            continue

        if minimum_value < 0:
            minimum_value = 0
        if maximum_value > 1:
            maximum_value = 1
        confidence = maximum_value - minimum_value
        consequent = '(data[\'errors\'] >= ' + str(minimum_value) + ' ) & (data[\'errors\'] <= ' + str(maximum_value) + ' )'
        df_part.loc[len(df_part),:] = [regras.iloc[n,0], consequent, minimum_value,maximum_value, confidence]

    return df_part

def get_support(regras, data):
    sup_list = []
    error_list = []
    coverage_list = []
    for n in range(len(regras)):
        rule = regras.loc[n, 'antecedent'] + ' & ' +  regras.loc[n, 'consequent']
        cov = set(data[eval(rule)].index)
        coverage_list.append(cov)
        sup = len(cov)/len(data)
        sup_list.append(sup)
        error_list.append(data[eval(rule)]['errors'].mean())
    regras['support'] = sup_list
    regras['average_error'] = error_list
    regras['coverage'] = coverage_list
    return regras

def summarize_descriptors(df_regras, features, df_name='data'):
    condition_lists = df_regras['antecedent'].str.split(' & ')
    subgroup_descriptions = []
    subgroup_dicts =[]
    for c in condition_lists:
        sg_dict = {x: {'min': float("-inf"),
                      'max': float("inf")} for x in features}
        for rule in c:
            if rule.find('<=') != -1:
                feature = rule.split(' <= ')[0][(rule.find("'") + 1):rule.replace("'", "$", 1).find("'")]
                value = rule.split(' <= ')[1].strip()[:-1]
                sg_dict[feature]['max'] = min(sg_dict[feature]['max'], float(value))
            elif rule.find('>') != 1:
                feature = rule.split(' > ')[0][(rule.find("'") + 1):rule.replace("'", "$", 1).find("'")]
                value = rule.split(' > ')[1].strip()[:-1]
                sg_dict[feature]['min'] = max(sg_dict[feature]['min'], float(value))
        descriptor_list = []
        subgroup_dict_list = []
        for f in features:
            if sg_dict[f]['min'] == float("-inf") and sg_dict[f]['max'] == float("inf"):
                continue
            elif sg_dict[f]['min'] == float("-inf"):
                descriptor_list.append(f'{f}<{sg_dict[f]["max"]}')
            elif sg_dict[f]['max'] == float("inf"):
                descriptor_list.append(f'{f}>={sg_dict[f]["min"]}')
            else:
                descriptor_list.append(f'{f}: [{sg_dict[f]["min"]}:{sg_dict[f]["max"]}[')
            sg_dict[f]['feature'] = f
            subgroup_dict_list.append(sg_dict[f])

        subgroup_descriptions.append(' AND '.join(descriptor_list))
        subgroup_dicts.append(subgroup_dict_list)
    return subgroup_descriptions, subgroup_dicts

def plot_lprs(data: pd.DataFrame,
              x_column: str,
              y_column: str,
              target: Iterable,
              subgroups: pd.DataFrame,
              plot_mean_error: bool = False,
              ax=None):
    """Plot a 2D scatterplot showing the samples, its classes, and the local performance regions passed as parameters
    to the function"""

    if ax is None:
        ax = plt.gca()

    sns.scatterplot(data=data,
                    x=x_column,
                    y=y_column,
                    hue=target,
                    s=20, alpha=0.5, ax=ax)
    # Rectangle displacement, estimated from my head
    delta = 0.02
    for lpr in subgroups.itertuples(index=False):
        # extract the subgroup limits
        rules = lpr.lpr_dict
        if len(rules) < 2:
            rule = rules[0]
            if isinstance(rule, dict):
                rule1_upperbound = rule["max"]
                rule1_lowerbound = rule["min"]
                rule1_attribute = rule["feature"]
                if rule1_lowerbound == float("-inf"):
                    rule1_lowerbound = data[rule1_attribute].min()
                if rule1_upperbound == float("inf"):
                    rule1_upperbound = data[rule1_attribute].max()
            if rule1_attribute == x_column:
                rule2_attribute = y_column
            else:
                rule2_attribute = x_column
            rule2_lowerbound = data[rule2_attribute].min()
            rule2_upperbound = data[rule2_attribute].max()
        else:
            rule1, rule2 = tuple(lpr.lpr_dict)
            if isinstance(rule1, dict):
                rule1_upperbound = rule1["max"]
                rule1_lowerbound = rule1["min"]
                rule1_attribute = rule1["feature"]
                if rule1_lowerbound == float("-inf"):
                    rule1_lowerbound = data[rule1_attribute].min()
                if rule1_upperbound == float("inf"):
                    rule1_upperbound = data[rule1_attribute].max()
            else:
                raise(NotImplementedError("I still can't deal with non numeric features!"))
            if isinstance(rule2, dict):
                rule2_upperbound = rule2["max"]
                rule2_lowerbound = rule2["min"]
                rule2_attribute = rule2["feature"]
                if rule2_lowerbound == float("-inf"):
                    rule2_lowerbound = data[rule2_attribute].min()
                if rule2_upperbound == float("inf"):
                    rule2_upperbound = data[rule2_attribute].max()
            else:
                raise(NotImplementedError("I still can't deal with non numeric features!"))
        # draw a red or green rectangle around the region of interest
        if lpr.mean_sg > lpr.mean_dataset:
            color = 'red'
        else:
            color = 'green'
        if rule1_attribute == x_column:
            ax.add_patch(plt.Rectangle((rule1_lowerbound - delta, rule2_lowerbound - delta),
                                       width=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       height=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            if plot_mean_error:
                ax.text(rule1_lowerbound, rule2_lowerbound, round(lpr.mean_sg, 4), fontsize=8)
            # ax.text(rule1_upperbound, rule2_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
        else:
            ax.add_patch(plt.Rectangle((rule2_lowerbound - delta, rule1_lowerbound - delta),
                                       width=rule2_upperbound - rule2_lowerbound + 2*delta,
                                       height=rule1_upperbound - rule1_lowerbound + 2*delta,
                                       fill=False, edgecolor=color, linewidth=1))
            if plot_mean_error:
                ax.text(rule2_lowerbound, rule1_lowerbound, round(lpr.mean_sg, 4), fontsize=8)
            # ax.text(rule2_upperbound, rule1_upperbound, round(subgroup.mean_dataset, 4), fontsize=8)
    return ax

def relevance(df_regras: pd.DataFrame, sample_index: int):
    max_error = 0
    for r in df_regras.itertuples(index=False):
        if sample_index in r.coverage:
            max_error = max(max_error, r.average_error)
    return max_error

def total_relevance(df_regras: pd.DataFrame, df_hard: pd.DataFrame):
    return df_hard.apply(lambda x: relevance(df_regras, x.index[0])).sum()

def summarize_lprs(df_regras: pd.DataFrame, budget: int, df_hard: pd.DataFrame):
    candidate = df_regras.copy()
    selected = pd.DataFrame(columns=candidate.columns)
    current_total_relevance = 0

    while len(selected) < budget:
        candidate['marg_rel'] = np.NaN
        for i in candidate.index:
            aux = selected.append(candidate.loc[i])
            candidate.loc[i, 'marg_rel'] = total_relevance(aux, df_hard) - current_total_relevance
        print(current_total_relevance)
        current_total_relevance = current_total_relevance + candidate['marg_rel'].max()
        selected_index = candidate['marg_rel'].idxmax()
        selected = selected.append(candidate.loc[selected_index])
        candidate = candidate.loc[candidate.index != selected_index]

    return selected


def extract_lprs(data: pd.DataFrame, support_threshold: float = 0.05, minimum_threshold: float = 0.18) -> pd.DataFrame:
    """
    Extracts Local Performance Regions (LPRs) based on decision tree rules using the given dataset.

    This function trains decision trees for each combination of features (individual and pairs),
    extracts the rules from the decision trees, and selects rules that meet the specified support
    and error thresholds. It then computes the corresponding LPRs and returns them as a DataFrame.

    Parameters:
    data (pd.DataFrame): The input dataset containing features and a target column named 'errors'.
    support_threshold (float): The minimum support threshold for the rules (default is 0.05).
    minimum_threshold (float): The minimum average error threshold for rules (default is 0.18).

    Returns:
    pd.DataFrame: A DataFrame containing the extracted LPRs with their associated antecedents,
                  consequents, support, average error, and other metrics.
    """
    df_regras = pd.DataFrame()
    min_samples = int(support_threshold * data.shape[0])
    # Rules with only one column
    for col in data.columns[:-1]:
        clf = DecisionTreeRegressor(max_depth=30,
                                    min_samples_split=20,
                                    min_samples_leaf=min_samples,
                                    random_state=57).fit(data[[col]], data.iloc[:, -1])  # Fits the tree
        regras = rules(clf, data[[col]])  # Extracts the rules generated by the tree

        df_part = get_max_min(regras, data[[col, 'errors']])
        df_part = get_support(df_part, data[[col, 'errors']])
        filtro1 = df_part['support'] >= support_threshold
        filtro2 = df_part['average_error'] >= minimum_threshold
        df_part = df_part.loc[filtro1 & filtro2]
        df_regras = pd.concat([df_regras, df_part], ignore_index=True)

    # Regras com 2 colunas cada
    for n, col1 in enumerate(data.columns[:-1]):
        for col2 in data.columns[n + 1:-1]:
            clf = DecisionTreeRegressor(max_depth=30,
                                        min_samples_split=20,
                                        min_samples_leaf=min_samples,
                                        random_state=57).fit(data[[col1, col2]], data.iloc[:, -1])
            regras = rules(clf, data[[col1, col2]])
            df_part = get_max_min(regras, data[[col1, col2, 'errors']])
            df_part = get_support(df_part, data[[col1, col2, 'errors']])
            filtro1 = df_part['support'] >= support_threshold
            filtro2 = df_part['average_error'] >= minimum_threshold
            df_part = df_part.loc[filtro1 & filtro2]
            df_regras = pd.concat([df_regras, df_part], ignore_index=True)
    return df_regras

def run_lpr_baseline(X_sd: pd.DataFrame, high_error: pd.Series, budget: int = 7):
    """Executes the Local Performance Regions (LPR) baseline analysis on input data.

    This function extracts LPRs from the input data, summarizes descriptors, and
    identifies regions with high error rates. It processes the data to generate
    a summary of LPRs that meet specified criteria.

    Args:
        X_sd (pd.DataFrame): Input DataFrame containing features and an 'errors' column
            representing error values for each sample.
        high_error (pd.Series): Boolean Series indicating samples with high error rates.
        budget (int, optional): Maximum number of LPRs to include in the summary. Defaults to 7.

    Returns:
        pd.DataFrame: A DataFrame containing summarized LPRs with columns for antecedents,
            consequents, support metrics, coverage information, and descriptive statistics.
            Only includes the top 7 most relevant LPRs for samples with high error rates.
    """
    df_regras = extract_lprs(X_sd)

    lpr_desc, lpr_dict = summarize_descriptors(df_regras, X_sd.columns[:-1], 'data')
    # adjusting coverage representation for further analyses
    df_regras['covered'] = df_regras['coverage'].apply(lambda x: X_sd.index.isin(x))
    df_regras['lpr_desc'] = lpr_desc
    df_regras['lpr_dict'] = lpr_dict
    df_regras['mean_sg'] = 1
    df_regras['mean_dataset'] = 0
    return summarize_lprs(df_regras, budget, X_sd.loc[high_error])