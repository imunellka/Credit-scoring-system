import pandas as pd
import streamlit as st
from PIL import Image
from model import open_data, preprocess_data, split_data, load_model_and_predict
def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/kredito24_card.png')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Credit scoring",
        page_icon=image,

    )
    st.image(image)
    st.divider()
    st.markdown(
        "## Определяем, вероятность просрочки выплат на 90 и более дней"
    , unsafe_allow_html=True)


def write_user_data(df):
    # st.write("## Ваши данные")
    # st.write(df)
    pass


def write_prediction(prediction, prediction_probas):
    col1,col2 = st.columns(2)
    col1.write("## Предсказание:")
    col1.write(f"<span style='color: #355179; font-size:28px;'>{prediction}</span>", unsafe_allow_html=True)

    if prediction == "Мы готовы оформить вам кредит":
        st.balloons()


    col2.write("## Вероятность предсказания:")
    alpha =prediction_probas["Вы не задерживали выплаты с вероятностью"].at[1]
    col2.write(f"<span style='color:#355179; font-size:28px;'>{round(alpha,2)}</span>", unsafe_allow_html=True)


def process_side_bar_inputs():
    st.sidebar.header('Задайте параметры')
    user_input_df = sidebar_input_features()
    train_df = open_data()
    # X_df, y_df = preprocess_data(train_df)
    # fit_and_save_model(X_df, y_df)

    train_X_df, _ = split_data(train_df)
    full_X_df = pd.concat((user_input_df, train_X_df), axis=0)
    preprocessed_X_df = preprocess_data(full_X_df, test=False)

    user_X_df = preprocessed_X_df[:1]
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():
    RevolvingUtilization = st.sidebar.slider("Общий баланс кредитных карт, деленный на лимит", min_value=0.0, max_value=1.3, value=1.0,
                            step=0.05)
    # ag = st.sidebar.slider("Возраст", min_value=0, max_value=110, value=50,
    #                         step=5)
    MonthlyIncom = st.sidebar.slider("Ежемесячный доход", min_value=0, max_value=300000, value=10000,
                                     step=10000)
    DebtRatio = st.sidebar.slider("Ежемесячные расходы", min_value=0, max_value=300000, value=10000,
                                  step=10000) / max(MonthlyIncom,1)
    RealEstateLoans = st.sidebar.selectbox("Сколько у вас было кредитов (в том числе авто)", ("до 2", "3-5", "6-13", "14-28", "более 29"))
    GroupAg = st.sidebar.selectbox("Выберите свою возрастную группу", ("до 18", "19-34", "35-59", "60-74", "старше 74"))
    NumberOfTime3059 = st.sidebar.slider("Сколько просрочек на 30-59 дней за 2 года", min_value=0, max_value=12, value=2,
                            step=1)
    NumberOfTime6089 = st.sidebar.slider("Сколько просрочек на 60-89 дней за 2 года", min_value=0, max_value=12, value=2,
                            step=1)
    NumberOfTimes90 = st.sidebar.slider("Сколько просрочек на 90+ дней за 2 года", min_value=0, max_value=12, value=2,
                                        step=1)
    # NumberOfOpen = st.sidebar.slider("Количество открытых кредитов и кредитных карт", min_value=0, max_value=10, value=9,
    #                                  step=1)
    # NumberOfDependent = st.sidebar.slider("Количество иждивенцев на попечении", min_value=0, max_value=25,
    #                                      value=2,
    #                                      step=1)
    transletion = {
        "до 18": "a",
        "19-34": "b",
        "35-59": "c",
        "60-74": "d",
        "старше 74": "e",
        "до 2": "a",
        "3-5": "b",
        "6-13": "c",
        "14-28": "d",
        "более 29": "e"
    }




    data = {
        "RevolvingUtilizationOfUnsecuredLines": RevolvingUtilization,
        #"age": ag,
        "NumberOfTime30-59DaysPastDueNotWorse": NumberOfTime3059,
        "DebtRatio": DebtRatio,
        #"MonthlyIncome": MonthlyIncom,
        #"NumberOfOpenCreditLinesAndLoans": NumberOfOpen,
        "NumberOfTimes90DaysLate": NumberOfTimes90,
        "NumberOfTime60-89DaysPastDueNotWorse": NumberOfTime6089,
        #"NumberOfDependents": NumberOfDependent,
        "RealEstateLoansOrLines": transletion[RealEstateLoans],
        "GroupAge": transletion[GroupAg]
    }

    df = pd.DataFrame(data, index=[0])
    return df


if __name__ == "__main__":
    process_main_page()
