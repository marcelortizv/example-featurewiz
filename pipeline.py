import pandas as pd

import numpy as np
import xgboost as xgb
from featurewiz import featurewiz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, classification_report


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

    data_path = 'data'
    filename_data = 'clean-dataset'

    print("Start reading dataset")
    data = pd.read_csv(f"{data_path}/{filename_data}.csv")
    print('This dataset has shape: ', data.shape)

    # pre processing data and scaling
    X, y = preprocess(data, 'dpnm', scaler=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=999)

    train_data = pd.concat([X_train, pd.Series(y_train)], axis=1)
    test_data = pd.concat([X_test, pd.Series(y_test)], axis=1)

    # rename targe column
    train_data.rename(columns={0: "dpnm"}, inplace=True)
    test_data.rename(columns={0: "dpnm"}, inplace=True)
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_data.dropna(subset=['dpnm'], inplace=True)
    test_data.dropna(subset=['dpnm'], inplace=True)

    print('Performing feature selection')
    trainm, testm = featurewiz(train_data,
                               target='dpnm',
                               test_data=test_data,
                               corr_limit=0.70,
                               verbose=2
                               )

    selected_features = testm.columns.tolist()
    print(f'Number of features selected: {len(selected_features)}')

    # training model
    print('Starting training:')
    num_round = 60
    params = {'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 0, 'random_state': 71}
    while True:
        xg_train = xgb.DMatrix(X_train[selected_features], label=y_train)
        watchlist = [(xg_train, 'train')]

        bst = xgb.train(params, xg_train, num_round, watchlist)
        scores = bst.get_score(importance_type='gain')
        scores_list = [(k, v) for k, v in scores.items()]
        scores_list.sort(key=lambda x: x[1], reverse=True)
        new_features = list(scores.keys())

        if len(selected_features) == len(new_features):
            print('___________________________________________________________________________')
            print('End training')
            break
        else:
            selected_features = new_features
    # TESTING PART
    X_test = X_test[selected_features]
    X_train = X_train[selected_features]
    xg_test = xgb.DMatrix(X_test, label=y_test)

    fpr, tpr, thresholds = roc_curve(y_train,
                                     bst.predict(xg_train))

    cutoff = thresholds[np.argmin(np.abs(fpr + tpr - 1))]
    print('The probability cutoff for prediction in model is: ', cutoff)

    y_pred = bst.predict(xg_test)
    ypredicted = np.where(y_pred > cutoff, 1, 0)
    print(classification_report(y_test, ypredicted, target_names=['No Fraud', 'Fraud']))

    print('Done')



