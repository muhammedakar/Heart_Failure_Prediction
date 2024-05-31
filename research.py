import numpy as np
import pandas as pd
from lib import EDA as e, Feng as f
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

e.set_display()

df = pd.read_csv('database/heart.csv')

result = e.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

for col in cat_cols:
    e.cat_summary(df, col)

e.plot_numerical_col(df, num_cols)

e.plot_categoric_col(df, cat_cols)

e.high_correlated_cols(df, num_cols, plot=True)

for col in cat_cols:
    e.target_summary_with_cat(df, 'HeartDisease', col)

e.rare_analyser(df, 'HeartDisease', cat_cols)

df['Cholesterol'].replace(0, np.nan, inplace=True)
df['RestingBP'].replace(0, np.nan, inplace=True)

e.missing_values_table(df)

df.dropna(subset=['RestingBP'], inplace=True)

df['AgeCat'] = pd.cut(df['Age'], [27, 45, 65, 78], labels=['Young', 'Middle', 'Senior'])
df['MaxHRCat'] = pd.cut(df['MaxHR'], [59, 100, 150, 300], labels=['Low', 'Moderate', 'High'])
df['RestingBPCat'] = pd.cut(df['RestingBP'], [79, 90, 120, 300], labels=['Low', 'Normal', 'High'])
df['OldpeakCat'] = pd.cut(df['Oldpeak'], [-2.65, 0.001, 1.5, 7], labels=['Low', 'Normal', 'High'])

df["Cholesterol"] = df["Cholesterol"].fillna(df.groupby(['AgeCat', 'Sex', 'RestingBPCat'])
                                             ["Cholesterol"].transform("mean"))

df.dropna(subset=['Cholesterol'], inplace=True)

df['CholesterolCat'] = pd.cut(df['Cholesterol'], [80, 200, 240, 280, np.inf],
                              labels=['Low', 'Normal', 'High', 'Very High'])

result = e.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

for col in cat_cols:
    e.target_summary_with_cat(df, 'HeartDisease', col)

for col in num_cols:
    f.check_outlier(df, col)

f.lof(df, plot=True)

lof_indexes = f.lof_indexes(df, 6)

df.drop(lof_indexes, inplace=True)

cat_cols = [col for col in cat_cols if col != 'HeartDisease']

df = f.one_hot_encoder(df, cat_cols, drop_first=True)

y = df["HeartDisease"]
X = df.drop(["HeartDisease"], axis=1)


def base_models(X, y):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
    print(score.T)


base_models(X, y)

knn_params = {"n_neighbors": range(2, 50), }

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(verbose=False), rf_params),
               (
                   'XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbose=False),
                   xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=0), lightgbm_params),
               ('CatBoost', CatBoostClassifier(verbose=False), catboost_params)
               ]


def hyperparameter_optimization(X, y, cv=3):
    print("Hyperparameter Optimization....")
    best_models = {}
    score = pd.DataFrame(index=['accuracy', 'f1', 'roc_auc'])
    for name, classifier, params in classifiers:
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)
        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=['accuracy', 'f1', 'roc_auc'])
        f1 = cv_results['test_f1'].mean()
        auc = cv_results['test_roc_auc'].mean()
        accuracy = cv_results['test_accuracy'].mean()
        score[name] = [accuracy, f1, auc]
        print(f'{name} hesaplandı...')
        best_models[name] = final_model
    print(score.T)
    return best_models


best_models = hyperparameter_optimization(X, y, cv=5)

best_params = best_models['CatBoost'].get_params()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
final_model = CatBoostClassifier(verbose=False, depth=3, iterations=500, learning_rate=0.01).fit(X, y)
y_pred = final_model.predict(X)
cv_results = cross_validate(final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
f1 = cv_results['test_f1'].mean()
auc = cv_results['test_roc_auc'].mean()
accuracy = cv_results['test_accuracy'].mean()

e.plot_importance(final_model, X)
f.plot_confusion_matrix(y, y_pred)
f.plot_roc_curve(final_model, X_test, y_test)

e.val_curve_params(final_model, X, y, param_name='iterations', param_range=[200, 500, 600, 700, 1000, 1500, 2500], cv=5)

joblib.dump(final_model, "deployment/final_model.pkl")
