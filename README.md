Based on https://github.com/SanzharAmirzhan/Kaggle-GiveMeSomeCredit

# credit_scoring

1) Create folder "models"
2) Create folder "data"
3) Download zip from https://www.kaggle.com/c/GiveMeSomeCredit/data and extract csv files to "data" folder


### Single row sample:

,SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents

,,0.88551908,43,0,0.177512717,5700,4,0,0,0,0

### Conda dependencies

conda install -c anaconda joblib<br/>
conda install -c anaconda numpy<br/>
conda install -c anaconda pandas<br/>
conda install -c anaconda seaborn<br/>
conda install -c conda-forge xgboost<br/>
conda install -c conda-forge lightgbm<br/>
conda install -c anaconda ipython<br/>