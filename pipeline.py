import pandas as pd

import numpy as np
from featurewiz import featurewiz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from featurewiz import simple_XGBoost_model


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def preprocess(df: pd.DataFrame, target_string: str, scaler: bool):
    """
    Process the raw dataset into to sets, predictors and target
    :param
    df: raw dataframe
    target_string: name of target columns
    scaler: True or False if you want to apply Scaling
    :return:
    X: predictors / features
    y: outcome / target
    """
    if target_string in list(df.columns):
        df = clean_dataset(df)
        y = df[str(target_string)].values
        X = df.drop([str(target_string)], axis=1)

        if scaler:
            sc = StandardScaler()
            X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

            return X, y

        return X, y
    else:
        raise ValueError("Error, target_string do not belong to this dataset")


if __name__ == '__main__':

    data_path = '../data'
    filename_data = 'clean-dataset'

    print("Start reading dataset")
    data = pd.read_csv(f"{data_path}/{filename_data}.csv")
    print('This dataset has shape: ', data.shape)

    # pre processing data and scaling
    X, y = preprocess(data, 'dpnm', scaler=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=999)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    print('Performing feature selection')
    trainm, testm = featurewiz(train=train_data,
                          target='dpnm',
                          test_data=test_data,
                          corr_limit=0.70,
                          verbose=2
                          )

    feats = testm.columns.tolist()
    print(f'Number of features selected: {len(feats)}')

    y_preds = simple_XGBoost_model(trainm[feats], trainm['dpnm'], testm[feats])
    y_pred = y_preds[0]

    print(balanced_accuracy_score(y_test, y_pred))
    print('Done')



