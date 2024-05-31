import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import validation_curve
import warnings
from sklearn.exceptions import ConvergenceWarning


def set_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 5000)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=SyntaxWarning)
    warnings.simplefilter("ignore", category=ConvergenceWarning)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)


def target_summary_with_num(dataframe, target, numerical_col):
    if len(dataframe[target].unique()) < 20:
        print(
            dataframe.groupby(target).agg(
                {numerical_col: ['min', 'max', "mean", 'median', 'std', 'skew', 'count', 'sum']}),
            end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    summary_df = dataframe.groupby(categorical_col).agg({
        target: ['min', 'max', 'mean', 'median', 'std', 'skew', 'count', 'sum']
    })
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df['survived_ratio'] = 100 * dataframe[categorical_col].value_counts() / len(dataframe)
    print(summary_df, end="\n\n\n")


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    binary_cols = [col for col in dataframe.columns if
                   dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'binary_cols: {len(binary_cols)}')

    return cat_cols, num_cols, cat_but_car, binary_cols


def missing_values_bar_grap(df):
    msno.bar(df)
    plt.show()


def missing_values_matrix_grap(df):
    msno.matrix(df)
    plt.show()


def missing_values_heatmap_grap(df):
    msno.heatmap(df)
    plt.show()


def plot_numerical_col(dataframe, num_cols, plot_type='hist'):
    num_cols_count = len(num_cols)
    num_rows = num_cols_count // 3
    num_rows += 1 if num_cols_count % 3 != 0 else 0  # Eğer sütun sayısı 3'e tam bölünmüyorsa bir ek satır oluştur.

    col_groups = [num_cols[i:i + 12] for i in range(0, num_cols_count, 12)]

    for group in col_groups:
        fig, axes = plt.subplots(num_rows, 3, figsize=(10, 10))
        axes = axes.flatten()

        for i, col in enumerate(group):
            if plot_type == 'hist':
                sns.histplot(data=dataframe[col], ax=axes[i])
            elif plot_type == 'kde':
                sns.kdeplot(data=dataframe[col], ax=axes[i])
            elif plot_type == 'box':
                sns.boxplot(data=dataframe[col], ax=axes[i])
            else:
                print("Geçersiz grafik türü. Lütfen 'hist', 'kde', veya 'box' olarak belirtin.")
                return
            axes[i].set_xlabel(col)

        for j in range(len(group), num_rows * 3):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def plot_categoric_col(dataframe, cat_cols):
    cat_cols_count = len(cat_cols)
    cat_rows = cat_cols_count // 3
    cat_rows += 1 if cat_cols_count % 3 != 0 else 0  # Eğer sütun sayısı 3'e tam bölünmüyorsa bir ek satır oluştur.

    fig, axes = plt.subplots(cat_rows, 3, figsize=(10, 10), squeeze=True)
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        sns.countplot(data=dataframe, x=col, ax=axes[i], order=dataframe[col].value_counts().index)
        axes[i].set_xlabel(col)

    plt.tight_layout()
    plt.show()


def high_correlated_cols(dataframe, num_cols, plot=False, corr_th=0.90):
    corr = dataframe[num_cols].corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        sns.set(rc={'figure.figsize': (12, 12)})
        colors = [(0, "darkgreen"), (0.5, "white"), (1, "darkblue")]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        sns.heatmap(corr, cmap=cmap, annot=True, fmt='.2f', linewidths=.5, cbar_kws={"shrink": .8})
        plt.show()

    return drop_list


def plot_importance(model, X, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len(X)])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=3):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)
