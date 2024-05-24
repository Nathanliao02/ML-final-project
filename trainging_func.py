from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_tune_and_evaluate_model(X, y, model, param_grid, standardize=True, resample=True, scoring='recall', cv=5, random_state=42):
    '''
    Train, tune, and evaluate a machine learning model.

    Parameters:
    - X (array-like): The input features.
    - y (array-like): The target variable.
    - model (estimator): The machine learning model to be trained and tuned.
    - param_grid (dict): The parameter grid for hyperparameter tuning.
    - standardize (bool): Whether to standardize the features. Default is True.
    - resample (bool): Whether to apply SMOTE resampling to the training data. Default is True.
    - scoring (str): The scoring metric to evaluate the model performance. Default is 'recall'.
    - cv (int): The number of cross-validation folds. Default is 5.
    - random_state (int): The random state for reproducibility. Default is 42.

    Returns:
    - best_model (estimator): The best trained model.
    - scores (dict): The evaluation scores of the best model.
    - best_params (dict): The best hyperparameters found during tuning.
    '''
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

    # Standardize features
    if standardize:
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)

    # Apply SMOTE to the scaled training data
    if resample:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # Combine the model and scaler into a single pipeline
    pipe = [('scaler', StandardScaler()), ('model', model)] if standardize else [('model', model)]
    pipeline = Pipeline(pipe)

    # Create the GridSearchCV object with StratifiedKFold
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, cv=stratified_cv)
    grid_search.fit(X_train, y_train)

    # Extracting the best estimator and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate on multiple metrics using the scaled test data
    y_pred = best_model.predict(X_test)
    y_scores = best_model.predict_proba(X_test)[:, 1]

    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_scores)
    }

    # Plotting the performance of various parameter combinations
    results = pd.DataFrame(grid_search.cv_results_)
    for param in param_grid:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results, x=f'param_{param}', y='mean_test_score')
        plt.title(f'Performance for different values of {param}')
        plt.ylabel(scoring)
        plt.xlabel(param)
        plt.show()

    print("Best Model:", best_model)
    print("Scores:", scores)
    print("Best Params:", best_params)

    return best_model, scores, best_params