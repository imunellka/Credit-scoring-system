# Credit-scoring-system
Интерактивные веб-приложение через streamlit для задачи кредитного скоринга

_Целевая переменная_ (таргет) – `SeriousDlqin2yrs`: клиент имел просрочку 90 и более дней

### Признаки
- `RevolvingUtilizationOfUnsecuredLines`: общий баланс средств (total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits)
- `age`: возраст заемщика
- `NumberOfOpenCreditLinesAndLoans`: количество открытых кредитов (напрмер, автокредит или ипотека) и кредитных карт
- `NumberOfTime30-59DaysPastDueNotWorse`: сколько раз за последние 2 года наблюдалась просрочка 30-59 дней
- `DebtRatio`: ежемесячные расходы (платеж по долгам, алиментам, расходы на проживания) деленные на месячный доход
- `MonthlyIncome`: ежемесячный доход
- `NumberOfTimes90DaysLate`: сколько раз наблюдалась просрочка (90 и более дней)
- `RealEstateLoansOrLines`: закодированное количество кредиов (в том числе под залог жилья) - чем больше код буквы, тем больше кредитов
- `NumberOfTime60-89DaysPastDueNotWorse`: сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней
- `NumberOfDependents`: количество иждивенцев на попечении (супруги, дети и др)
- `GroupAge`: закодированная возрастная группа - чем больше код, тем больше возраст

### Особенности очистки датасета
В ходе анализа распределений данных, перцентилей (min, 25%, 75%, max) и ящиков с усами, опираясь также на знания из реальной математики.
В данных были найдены неккоректные значения, так как данных у нас достаточно много, а выбросов по сравнению с ними неумолимо мало было принято решение удалить выбросы. (Благодаря этому изменения ушла сильная корреляция между некоторыми регрессорами и улучшилась корреляция регрессоров с таргет-столбцом)

Корректировка задела следующие столбцы:
- `RevolvingUtilizationOfUnsecuredLines`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `NumberOfTimes90DaysLate`
- `NumberOfTime60-89DaysPastDueNotWorse`

Тагрет-стоблец имеет сильный дисбаланс классов => как метрику будем использовать f1_score (precision, recall)

В остальном без особенностей ( заполнение пустых значений, кодирование категориальных переменных - порядковым способом, тк переменные взаимосвязанны)

### EDA (Разведочный анализ) 

- Посмотрели на графики распределения переменных
- Оценили таблицу корреляции ( эмперические наблюдения о будущей "успешности" модели)
- Посмотрели на диаграммы рассеивания
- На графиках рассмотрели связь с закодированным количеством кредитов, возрастными группами, количеством иждивенцев на попечении
