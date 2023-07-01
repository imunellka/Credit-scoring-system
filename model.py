from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from pickle import dump, load
import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame):
    y = df['SeriousDlqin2yrs']
    X = df[["RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio","NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "RealEstateLoansOrLines","GroupAge"]]
    return X, y


def open_data(path="data/credit_scoring.csv"):
    df = pd.read_csv(path)
    df = df[['SeriousDlqin2yrs', "RevolvingUtilizationOfUnsecuredLines", "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio","NumberOfTimes90DaysLate",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "RealEstateLoansOrLines","GroupAge"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    #df.dropna(inplace=True)
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 1.35]
    df = df[df['RevolvingUtilizationOfUnsecuredLines'] >= -0.76]
    df = df[df["NumberOfTime30-59DaysPastDueNotWorse"] <= 24]
    df = df[df["NumberOfTimes90DaysLate"] <= 24]
    df = df[df["NumberOfTime60-89DaysPastDueNotWorse"] <= 24]
    df = df[df['DebtRatio'] <= 1.9]
    df = df[df['DebtRatio'] >= -0.86]
    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df
    #X_df.age = X_df.age.fillna(0)
    #X_df.MonthlyIncome = X_df.MonthlyIncome.fillna(X_df.MonthlyIncome.median())
    #X_df.NumberOfDependents = X_df.NumberOfDependents.fillna(1)
    translate = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    X_df.RealEstateLoansOrLines = X_df.RealEstateLoansOrLines.apply(lambda x: x.lower())
    X_df.RealEstateLoansOrLines = np.where(np.isin(X_df.RealEstateLoansOrLines, list(translate.keys())),
                                           [translate[feature.lower()] for feature in X_df.RealEstateLoansOrLines],
                                           X_df.RealEstateLoansOrLines)
    X_df.GroupAge = np.where(np.isin(X_df.GroupAge, list(translate.keys())),
                             [translate[feature.lower()] for feature in X_df.GroupAge], X_df.GroupAge)
    X_df.RealEstateLoansOrLines = X_df.RealEstateLoansOrLines.astype('int')
    X_df.GroupAge = X_df.GroupAge.astype('int')
    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path="data/saved_model.mw"):
    model = RandomForestClassifier(n_estimators=500, criterion="entropy", min_samples_leaf=2, max_features="sqrt",
                                    bootstrap=True)
    #X_df = X_df.drop(columns=['MonthlyIncome', 'NumberOfDependents','NumberOfOpenCreditLinesAndLoans', 'age'])
    model.fit(X_df, y_df)
    test_prediction = model.predict(X_df)
    f1 = f1_score(test_prediction, y_df)
    print(f"Model f1_score is {f1}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/saved_model.mw"):
    with open(path, "rb") as file:
        try:
            model = load(file)
        except:
            print(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вы не задерживали выплаты с вероятностью"
    }

    encode_prediction = {
        1: """Простите,
        <br> вероятность просрочки слишком высока""",
        0: "Мы готовы оформить вам кредит"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[1])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)