import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from statsmodels.stats.proportion import proportions_ztest
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
import graphviz
import pydotplus


def outlier_thresholds(dataframe, column, q1=0.05, q3=0.95):
    """
    Aykiri degerlerin alt limitini ve ust limitini donduren fonksiyon
    Parameters
    ----------
    dataframe: Pandas.series
        Aykiri degerin bulunmasi istediginiz dataframe'i giriniz
    column: str
        Hangi degisken oldugunu belirtiniz
    q1:  float
        Alt limit ceyrekligini belirtin
    q3: float
        Ust limit ceyrekligini belirtin

    Returns
    -------
    low_th: float
        alt eşik değer
    up_th: float
        üst eşik değer
    """
    quartile1 = dataframe[column].quantile(q1)
    quartile3 = dataframe[column].quantile(q3)
    iqr = quartile3 - quartile1
    up_th = quartile3 + 1.5 * iqr
    low_th = quartile1 - 1.5 * iqr
    return low_th, up_th


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    if not outliers.empty:
        return print(col_name, True)
    else:
        return print(col_name, False)


def grab_outliers(dataframe, col_name, index=False):
    """
    Bu fonksiyon bize ilgili değişkendeki aykırı değerleri döndürür.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        Hangi değişkenin bilgilerini istiyorsanız o değişkeni yazınız
    index: bool
        Fonksiyonun geriye aykırı değerleri dönmesini istiyorsanız True verin


    Returns
    -------

    """
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_specific_outlier(dataframe, col_name):
    """
    Bu fonskiyon istediğmiz kolondaki aykırı değerleri siler
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col_name: str
        dataframe deki aykırı değerleri silmek istediğimiz değişkenin ismi

    Returns
    -------
        df_without_outliers: dataframe
            Aykırı değerlerin silindiği dataframe i döner

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, col):
    """
    Belli bir değişkendeki aykırı değerleri baskılamak yerini alt ve üst limitlerle doldurmak için kullanınız.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    col: str
        Baskılamak istediğiniz değişkenin ismi

    Returns
    -------
        dataframe: dataframe
            Baskılanmış yerini alt ve üst limitlerle doldurulmuş bir dataframe döndürür

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col)
    dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
    dataframe.loc[(dataframe[col] > up_limit), col] = up_limit


def lof(dataframe, neighbors=20, plot=False):
    """
    İstenilen veri setine LOF (Local Outlier Factor) yöntemini uygular.
    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenilen dataframe
    th_index: int
        İçerisine index bilgisi alır o indeksten sonraki verileri kırpar eğer bilgi girilmezse istenilen veriler getirilmez
    neighbors: int
        Komşuluk sayısını belirtiniz.

    Returns
    -------
        df_scores: np.Array
            Geriye değişkenlerin skorlarını döndürür.
    """
    dataframe = dataframe.select_dtypes(include=['float64', 'int64'])
    clf = LocalOutlierFactor(n_neighbors=neighbors)
    clf.fit_predict(dataframe)
    df_scores = clf.negative_outlier_factor_
    if plot:
        scores = pd.DataFrame(np.sort(df_scores))
        scores.plot(stacked=True, xlim=[0, 50], style='.-')
        plt.show()

    return df_scores


def lof_indexes(df, threshold):
    df_scores = lof(df)
    th = np.sort(df_scores)[threshold]
    return df[df_scores < th].index


def label_encoder(dataframe, binary_col, info=False):
    labelencoder = LabelEncoder()

    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    if info:
        d1, d2 = labelencoder.inverse_transform([0, 1])
        print(f'{binary_col}\n0:{d1}, 1:{d2}')
    return dataframe


def encode_all_binary_columns(dataframe, binary_cols, info=False):
    for col in binary_cols:
        label_encoder(dataframe, col, info)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def hypothesis_testing(df, new_feature, target):
    test_stat, pvalue = proportions_ztest(count=[df.loc[df[new_feature] == 1, target].sum(),
                                                 df.loc[df[new_feature] == 0, target].sum()],

                                          nobs=[df.loc[df[new_feature] == 1, target].shape[0],
                                                df.loc[df[new_feature] == 0, target].shape[0]])

    print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


def plot_roc_curve(log_model, X_test, y_test):
    RocCurveDisplay.from_estimator(log_model, X_test, y_test, plot_chance_level=True)
    plt.title('ROC Curve')
    plt.show()


def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)
