import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from math import sqrt
from PIL import Image
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, RFECV
from sklearn.linear_model import LinearRegression
pd.set_option('display.max_columns', None)

cases_state = pd.read_csv('cases_state.csv')
tests_state = pd.read_csv('tests_state.csv')
tests_malaysia = pd.read_csv('tests_malaysia.csv')
cases_malaysia = pd.read_csv('cases_malaysia.csv')
clusters = pd.read_csv('clusters.csv')
population = pd.read_csv('population.csv')
cases_malaysia['date'] = pd.to_datetime(cases_malaysia['date'], errors='raise')
tests_malaysia['date'] = pd.to_datetime(tests_malaysia['date'], errors='raise')
cases_state['date'] = pd.to_datetime(cases_state['date'], errors='raise')
tests_state['date'] = pd.to_datetime(tests_state['date'], errors='raise')
cases_malaysia.drop(cases_malaysia.columns[4:], axis=1, inplace=True)
cases_malaysia.drop(cases_malaysia[cases_malaysia['date'] > '2021-09-10'].index, inplace=True)
tests_malaysia.drop(tests_malaysia[tests_malaysia['date'] > '2021-09-10'].index, inplace=True)
cases_malaysia.set_index('date', inplace=True)
tests_malaysia.set_index('date', inplace=True) 
merged_malaysia = pd.merge(left=cases_malaysia, right=tests_malaysia, how='inner', 
                           left_index=True, right_index=True)

st.title("Good day to you, dear reader!")
st.header("These are the findings for our group assignment!")

st.header("Question 3 (i)")
st.subheader('Our Exploratory Data Analysis')
st.markdown('Our EDA will first cover the data for Malaysia, followed by the states in Malaysia')
st.markdown('')

st.subheader("Malaysia")
st.markdown("Below is the correlation matrix for the new cases, imported cases, recovered cases, tests using RTK_AG and PCR")
corr_mat = alt.Chart(merged_malaysia).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='Origin:N'
).properties(
    width=100,
    height=100
).repeat(
    row=['cases_new', 'cases_import', 'cases_recovered', 'rtk-ag', 'pcr'],
    column=['cases_new', 'cases_import', 'cases_recovered', 'rtk-ag', 'pcr']
)
st.altair_chart(corr_mat)

st.markdown('From visually inspecting the scatter plot matrix above, we can see that cases_new and cases_recovered are similar in their distribution, that is, centering around the lower end of 1000 cases per day and higly skewed to the right. A parallel observation can be seen for rtk-ag and pcr. We can also make the safe assumption that there many upper outliers for all the variables based on the skewness of the distributions. The scatter plots show that, with the exception of cases_import, the variables are have a positive association with another.')
st.markdown('')

st.markdown('Below is the relational plot between daily new cases and cases recovered.')
Image.open('relplot.JPG').convert('RGB').save('relplot.jpeg')
im = Image.open("relplot.jpeg")
st.image(im, width=800, caption="The new cases and cases recovered relational plot")
st.markdown('')

st.markdown('Below is the relational plot for the tests done using PCR and RTK-AG.')
Image.open('relplot_tests_malaysia.JPG').convert('RGB').save('relplot_tests_malaysia.jpeg')
im = Image.open("relplot_tests_malaysia.jpeg")
st.image(im, width=800, caption="The relational plot for PCR and RTK-AG")
st.markdown('')

st.subheader('States in Malaysia')
st.markdown('Below is the relational plot for daily cases in each state.')
Image.open('relplot_cases_states.JPG').convert('RGB').save('relplot_cases_states.jpeg')
im = Image.open("relplot_cases_states.jpeg")
st.image(im, width=800, caption="The relational plot for total cases in each state")
st.markdown('')

st.header("Question 3 (ii)")
st.subheader('We need to identify the states with the strongest correlation to Pahang and Johor')
st.markdown('We will split the data into 4 different correlation matrices. The first and second matrice roughly corresponds to the first and second wave of Covid-19 in Malaysia. The third and fourth matrice represents the third wave as a whole.')

def get_df_stateandcases(df):

    #df_toMerge = df
    df_temp = df.copy()
    df_temp = df_temp.loc[:, ['date', 'cases_new']]

    # https://note.nkmk.me/en/python-pandas-dataframe-rename/
    df_temp=df_temp.rename(columns={'cases_new': 'Johor_cases'})
    
    df_toMerge = df_temp.copy()
    
    state_list = ['Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan',
       'Pahang', 'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak',
       'Selangor', 'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan',
       'W.P. Putrajaya']
    
    for x in state_list:
        df_toMerge = combine_state(df, df_toMerge, x)
        #print(df_toMerge)
        
    #print(df_toMerge)
    return df_toMerge
        
def combine_state(df, df_toCombine, state_name):
    df_temp = df.loc[df['state'] == state_name]
    df_temp = df_temp.loc[:, ['date', 'cases_new']]
    column_name = state_name + '_cases'
    df_temp=df_temp.rename(columns={'cases_new': column_name})

    #merged_malaysia = pd.merge(left=case_malaysia, right=tests_malaysia, how='left', left_on=['date'], right_on=['date'])
    df_combined = pd.merge(left=df_toCombine, right=df_temp, how='left', left_on=['date'], right_on=['date'])
    
    return df_combined

df_state_cases = get_df_stateandcases(cases_state)
df_split_4_portion = np.array_split(df_state_cases, 4)

for i in range(0,4):
    st.markdown('Time Period ' + str(i+1))
    st.markdown("Date: " + str(df_split_4_portion[i]['date'].min()) + ' - ' + str(df_split_4_portion[i]['date'].max()))
    cor_data = (df_split_4_portion[i]
                  .corr().stack()
                  .reset_index()     # The stacking results in an index on the correlation values, we need the index as normal columns for Altair
                  .rename(columns={0: 'correlation', 'level_0': 'state1', 'level_1': 'state2'}))
    cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)  # Round to 2 decimal
    # cor_data

    base = alt.Chart(cor_data).encode(
        x='state1:O',
        y='state2:O'    
    )

    # Text layer with correlation labels
    # Colors are for easier readability
    text = base.mark_text().encode(
        text='correlation_label',
    )

    # The correlation heatmap itself
    cor_plot = base.mark_rect().encode(
        color='correlation:Q'
    ).properties(height=700, width=700)

    cor_plot + text

def calc_lag(df, ndays):
    """Takes in cases dataset for a particular state and number of days. 
    Returns a new dataset with added n-day columns of 1 to n-day lags for each of the
    3 cases columns."""
    
    cols_of_interest = df[['cases_import', 'cases_new', 'cases_recovered']]
    for i in range(ndays):
        current_day_lag = cols_of_interest.shift(i+1)
        current_day_lag.columns = [str(i+1)+'-day lag imported case', 
                                   str(i+1)+'-day lag new case', 
                                   str(i+1)+'-day lag recovered case']
        df = pd.concat([df, current_day_lag], axis=1)
    return df

def featurize_cases(df_cases_state, state_name):
    """Takes in the cases_state dataset and a name of a state. Returns a new dataset filtered for
    this state that contains added features/columns.
    """
    
    # extracting dataframe for this state
    cases = df_cases_state[df_cases_state['state'] == state_name]
    cases = cases.drop(columns=['state'])
    cases.reset_index(inplace=True, drop=True)

    # adding columns of the rolling averages of new cases for 7 days and 14 days
    cases['7DayAverage'] = cases['cases_new'].rolling(window=7).mean()
    cases['14DayAverage'] = cases['cases_new'].rolling(window=14).mean()

    # adding columns of 1-to-n-day-lags of new cases, new import cases and cases recovered
    cases = calc_lag(cases, 21)

    # simple plot to confirm...
    # plt.figure(figsize = (15,8))
    # df = cases.set_index('date')
    # sns.lineplot(data=df.loc['2021-01':,['cases_new', 7DayAverage', '14DayAverage']])

    # dropping null value rows as a result of the n-day average columns
    res = cases.dropna()
    
    # finally, remove rows after cut-off date
    res.drop(res[res['date'] > '2021-09-10'].index, inplace=True)
    
    return res

def feature_extraction(df, state):
    """Takes in cases dataset and a state name. Extracts and returns the best selected
    features for this state's data, and plots the top 7 features."""
    
    X = df.drop(columns=['cases_new','date'], axis=1)
    y = df['cases_new']
    
    chi2_extractor = SelectKBest(chi2, k=5)
    chi2_extractor.fit(X, y)
    chi2_features = X.columns[chi2_extractor.get_support()]
    chi2_features = chi2_features.tolist()
    
    fregres_extractor = SelectKBest(f_regression, k=5)
    fregres_extractor.fit(X, y)
    fregres_features = X.columns[fregres_extractor.get_support()]
    fregres_features = fregres_features.tolist()
    
    minforegress_extractor = SelectKBest(mutual_info_regression, k=5)
    minforegress_extractor.fit(X, y)
    minforegress_features = X.columns[minforegress_extractor.get_support()]
    minforegress_features = minforegress_features.tolist()
    
    rfecv_extractor = RFECV(LinearRegression(), min_features_to_select=5)
    rfecv_extractor.fit(X, y)
    rfecv_features = X.columns[rfecv_extractor.get_support()]
    rfecv_features = rfecv_features.tolist()
    
    feature_list = list(chain(rfecv_features, chi2_features, minforegress_features, fregres_features))
    d = Counter(feature_list)
    
    feature_table = pd.DataFrame.from_dict(d, orient='index').reset_index()
    feature_table = feature_table.rename(columns={'index':'features_extracted', 0:'count'})
    feature_table = feature_table.sort_values('count', ascending=False)
    feature_table
    
    bar = alt.Chart(feature_table).mark_bar(size=10).encode(
        x='count',
        y='features_extracted',
        color=alt.condition(
            alt.datum.count >= 3,  
            alt.value('red'),     
            alt.value('steelblue')   
        )
    ).properties(height=300, width=700)
    
    st.altair_chart(bar)

    # return list of top features given some min count
    best_features_filt = feature_table['count'] >= 3
    return feature_table.loc[best_features_filt, 'features_extracted'].values.tolist()

st.markdown('From the last correlation heatmap, we can see that Pahang has strong correlation (>0.8) to Kedah, Perak, Pulau Pinang, Sabah and Terengganu. Meanwhile Johor does not have strong correlation (>0.8) to any of the states.')

chosen_states = ['Pahang', 'Kedah', 'Johor', 'Selangor']

# get the featurized cases for each state, store in dict
state_featurized = {state: featurize_cases(cases_state, state) for state in chosen_states} 

# merging with tests_state data
st.header('Question 3 (iii)')
st.subheader('The question requires us to identify strong features/indicators to daily cases')
st.markdown("The option below allows you to set whether to merge the PCR and RTK-AG as part of the features to be extracted. Different charts will be generated depending on your selected option.")
st.markdown('The differences are: ')
st.markdown('Dates: If merged, the date starts from 2021-07-01. Else, the date starts from 2020-01-25')
st.markdown('Features: If merged, the PCR and RTK-AG data are included for feature extraction process. Else, they are not included.')

st.markdown("")

option = st.selectbox(
    'Would you like to merge the PCR and RTK-AG for feature extraction?',
     ['Yes', 'No'])
if option == 'Yes':
    bool_merge = True
elif option == 'No':
    bool_merge = False
    
for state in chosen_states:
    # extracting tests data for this state
    tests = tests_state[tests_state['state'] == state]
    tests = tests.drop(columns=['state'])
    tests.reset_index(inplace=True, drop=True)

    # left merge tests and featurized cases on date
    res = pd.merge(left=tests, right=state_featurized[state], how='left')
    
    # dropping null value rows
    res = res.dropna()
    
    # finally, remove rows after cut-off date
    res.drop(res[res['date'] > '2021-09-10'].index, inplace=True)
    
    if bool_merge:
        state_featurized[state] = res

# dict to hold the best features for each state
best_features = {state: [] for state in chosen_states}

# plot and extract
st.text("Note: A red bar represents a feature that was selected in 3 or more feature selectors.")
for i, state in enumerate(chosen_states):
    st.text("These are the features extracted for " + chosen_states[i])
    best_features[state] = feature_extraction(state_featurized[state], state)


st.header("Question 3 (iv) ")
st.subheader("Concerning the prediction of the daily cases in Pahang, Kedah, Johor, and Selangor, below are our findings based on classifcation and regression models.")

st.text('')

st.subheader("Classification Models")

def model_building_and_accuracy(df, state):

    df['danger'] = pd.cut(df['cases_new'], bins=[-1, 500, 1500, 3000, 100000], labels=['low', 'medium_low', 'medium_high', 'high'])
    
    X = df.loc[:, best_features[state]].values # using best features as predictors
    y = df.loc[:, 'danger'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    train_acc = pd.Series({'Decision Tree Classifier': 0.0, 'SVM': 0.0,
                      'GaussianNB': 0.0})
    test_acc = pd.Series({'Decision Tree Classifier': 0.0, 'SVM': 0.0,
                     'GaussianNB': 0.0})
    data_acc = pd.DataFrame({'Train accuracy (%)':train_acc, 'Test accuracy (%)':test_acc})
    
    model_DT = DecisionTreeClassifier(max_depth=3)
    model_DT.fit(X_train, y_train)
    y_pred = model_DT.predict(X_test)
    
    data_acc['Train accuracy (%)']['Decision Tree Classifier'] = model_DT.score(X_train, y_train)
    data_acc['Test accuracy (%)']['Decision Tree Classifier'] = model_DT.score(X_test, y_test)
    
    #Import svm model
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear', gamma='auto') # Linear Kernel

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    
    data_acc['Train accuracy (%)']['SVM'] = clf.score(X_train, y_train)
    data_acc['Test accuracy (%)']['SVM'] = clf.score(X_test, y_test)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    
    confusion_majority=confusion_matrix(y_test, y_pred)
    
    data_acc['Train accuracy (%)']['GaussianNB'] = nb.score(X_train, y_train)
    data_acc['Test accuracy (%)']['GaussianNB'] = nb.score(X_test, y_test)
    
    return data_acc

for i, state in enumerate(chosen_states):
    st.text("These are the training and test accuracy for " + chosen_states[i])
    df_temp = model_building_and_accuracy(state_featurized[state], state)
    df_temp

st.subheader("Regression Models")
st.markdown("Linear Regression")
for i, state in enumerate(chosen_states):
    # defining predictors and target values
    x = state_featurized[state].loc[:, best_features[state]].values # using best features as predictors
    y = state_featurized[state].loc[:, 'cases_new'].values

    # splitting data values into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # training model
    lin_model = LinearRegression()
    lin_model.fit(x_train, y_train)

    # predicting results
    y_pred = lin_model.predict(x_test)

    # evaluating model
    print(f'Linear Regression model evaluation for {state}:')
    print('\tR-square =', r2_score(y_test, y_pred))
    print('\tmean-squared-error =', sqrt(mean_squared_error(y_test, y_pred)))
    print('')

    # time-series best fitting curve plot
    cases_new_compare = state_featurized[state].loc[:, ['date', 'cases_new']]
    cases_new_compare['cases_new_pred'] = lin_model.predict(x)
    st.markdown(state)
    line = alt.Chart(cases_new_compare).transform_fold(
        ['cases_new', 'cases_new_pred'],
    ).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='key:N'
    ).properties(height=300, width=700)
    st.altair_chart(line)
    
st.markdown('')
st.markdown("Decision Tree Regression")
for i, state in enumerate(chosen_states):
    # defining predictors and target values
    x = state_featurized[state].loc[:, best_features[state]].values # using best features as predictors
    y = state_featurized[state].loc[:, 'cases_new'].values

    # splitting data values into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    # training model
    regr = DecisionTreeRegressor(random_state=0)
    regr.fit(x_train, y_train)

    # predicting results
    y_pred = regr.predict(x_test)

    # evaluating model
    print(f'Decision Tree Regression model evaluation for {state}:')
    print('\tTraining score = ', regr.score(x_train, y_train))
    print('\tTesting score = ', regr.score(x_test, y_test))
    print('\tR-square =', r2_score(y_test, y_pred))
    print('\troot-mean-squared-error =', sqrt(mean_squared_error(y_test, y_pred)))
    print('')
    # sns.relplot(x=y_test, y=y_pred, kind='scatter')


    # time-series best fitting curve plot
    cases_new_compare = state_featurized[state].loc[:, ['date', 'cases_new']]
    cases_new_compare['cases_new_pred'] = regr.predict(x)
    st.markdown(state)
    line = alt.Chart(cases_new_compare).transform_fold(
        ['cases_new', 'cases_new_pred'],
    ).mark_line().encode(
        x='date:T',
        y='value:Q',
        color='key:N'
    ).properties(height=300, width=700)
    st.altair_chart(line)