import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('https://raw.githubusercontent.com/VithalAgrawal/CustomerChurnPredictionDashboard/main/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.drop(labels=['customerID'], axis=1, inplace=True)

#Converting Total charges to numeric format
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

#Total charges has 11 rows with null values so we will drop them
df.dropna(inplace = True)


#converting all variables to numeric format
df_num = df.copy()
cols = ['Partner','Dependents','PhoneService','DeviceProtection','TechSupport','StreamingTV',
        'StreamingMovies','OnlineSecurity','OnlineBackup','MultipleLines','InternetService', 'Contract', 'PaymentMethod']
df_num = pd.get_dummies(df_num, columns=cols)
df_num['Churn'].replace({'No':0, 'Yes':1}, inplace = True)
df_num['PaperlessBilling'].replace({'No':0, 'Yes':1}, inplace = True)
df_num['gender'].replace({'Female':0, 'Male':1}, inplace = True)

for column in df_num.columns:
    df_num[column] = (df_num[column] - df_num[column].min()) / (df_num[column].max() - df_num[column].min())


X = df_num.drop('Churn', axis=1)
y = df_num['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LogisticRegression()

rf = RandomForestClassifier(class_weight = 'balanced_subsample', n_estimators=150, max_depth=6)

svm = SVC()

dt = DecisionTreeClassifier(max_depth = 7)

xgb = XGBClassifier(n_estimators=500, eta=0.01, subsample=0.5, colsample_bytree=0.5)

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
adb = AdaBoostClassifier(estimator=logreg, n_estimators=100, learning_rate=0.13)

vclf = VotingClassifier(estimators=[('RF', rf), ('XGB', xgb), ('ADB', adb)],voting='soft', weights=[1,3,3])

lr.fit(X_train, y_train)
y_pred_logreg = lr.predict(X_test)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

adb.fit(X_train, y_train)
y_pred_adb = adb.predict(X_test)

vclf.fit(X_train, y_train)
y_pred_vclf = vclf.predict(X_test)

models = {
    'Logistic Regression': y_pred_logreg,
    'Random Forest': y_pred_rf,
    'Support Vector Machine': y_pred_svm,
    'Decision Tree': y_pred_dt,
    'XGBoost': y_pred_xgb,
    'AdaBoost': y_pred_adb,
    'Voting Classifier': y_pred_vclf
}

# Calculate the evaluation metrics for each model
accuracy = []
precision = []
recall = []
f1 = []

for model_name, y_pred in models.items():
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))


# app = dash.Dash(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])


# defining the layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='TELECOM CUSTOMER CHURN DASHBOARD', style={'textAlign': 'center', 
                                                                # 'color': '#f0fff0',
                                'font-size': 50, 'margin':'20px'}),

    html.Hr(style={'margin':'20px 0 40px 0'}),

    html.H3(children='Exploratory Data Analysis:', style={'textAlign': 'center', 
                                            #    'color': '#f0fff0',
                                'font-size': 40, 'margin':'20px'}),

    html.Div([
                html.Div([
                    # html.H2(children='SunBurst chart of churn counts by gender', style={'textAlign': 'center', 'color': '#f0fff0',
                    #             'font-size': 30}),
                    # adding a pie chart of churn counts by gender
                    html.Div(dcc.Graph(
                        id='churn-by-gender-pie',
                        figure={}))
                ], style={'width': '45%'}),
                html.Div([
                    # html.H2(children='SunBurst chart of churn counts by senior citizen', style={'textAlign': 'center', 'color': '#f0fff0',
                    #                 'font-size': 30}),
                    # adding a pie chart of churn counts by senior citizen
                    html.Div(dcc.Graph(
                        id='churn-by-SeniorCitizen-sunburst',
                        figure={}))
                ], style={'width': '45%'})
            ], style={'display': 'flex', 'justify-content': 'space-around', 'margin-bottom':'40px'}),

    html.Div([
        dcc.Graph(
                id='corr-graph',
                figure={}),
        ], style={'margin': '10px auto', 'height':'450px', 'width':'90%'}),

    html.Hr(style={'margin':'60px auto 30px auto'}),

    html.Div([
            html.H3(children='Comparative Analysis Of All Models:', style={'textAlign': 'center', 
                                                                        #    'color': '#f0fff0', 
                                                                           'margin':'20px auto 10px auto',
                                    'font-size': 30}),
            # add an ROC curve
            dcc.Graph(
                id='comp_fig',
                figure={}),
        ], style={'height':'500px', 'margin':'30px auto'}),

    html.Hr(style={'margin':'20px 0 40px 0'}),



    html.Div([
                # Creating a division for adding dropdown helper text for choosing model
                    html.Div([
                                html.H3('To try out different models, choose model-', 
                                        style={
                                            'margin-right': '1.5rem', 
                                            'font-size': 30}
                                            )
                            ]),
                    dcc.Dropdown(id='model-dropdown',options=[
                                    {'label': 'Logistic Regression', 'value': 'lr'},
                                    {'label': 'Random Forest', 'value': 'rf'},
                                    {'label': 'Decision Tree', 'value': 'dt'},
                                    {'label': 'Support Vector Machine', 'value': 'svc'},
                                    {'label': 'XGBoost', 'value': 'xgb'},
                                    {'label': 'AdaBoost', 'value': 'adb'},
                                    {'label': 'Voting Classifier', 'value': 'vclf'},
                            ], value='lr', style={'width': '50%', 'padding': '3px', 'font-size': '20px', 'text-align-last': 'center', 'color':'black'}),
                                                    
            # Placing them next to each other using the division style flex
            ], style={'display': 'flex', 'justify-content': 'center'}),

    html.Hr(style={'margin':'20px 0 0 0'}),

    html.Div([
        html.Div([
            html.H3(children='Confusion Matrix', style={'textAlign': 'center', 
                                                        # 'color': '#f0fff0',
                                    'font-size': 30}),
            # adding a confusion matrix of predicted vs. actual values
            dcc.Graph(
                id='predicted-vs-actual-confusion-matrix',
                figure={}),
        ], style={'width':'90%', 'margin':'50px'}),
        html.Div([
            html.H3(children='ROC Curve', style={'textAlign': 'center', 
                                                #  'color': '#f0fff0',
                                    'font-size': 30}),
            # adding an ROC curve
            dcc.Graph(
                id='roc-curve',
                figure={}),
        ], style={'width':'90%', 'margin':'50px'})
    ], style={'display': 'flex', 'justify-content': 'space-around'}),

    html.Div([
        html.Div([
            html.H3(children='Classification Report', style={'textAlign': 'center', 
                                                            #  'color': '#f0fff0',
                                    'font-size': 30}),
            # adding a classification report showing accuracy, precision, recall and f1 scores
            dcc.Graph(
                id='classification-report-bar-fig',
                figure={}),
        ], style={'width':'50%', 'margin':'10px auto'}),
    ], style={'textAlign': 'center', 'width':'100%', 'margin':'10px auto'}),
])

# add callback functions to update the charts based on model selection
@app.callback(
    Output('predicted-vs-actual-confusion-matrix', 'figure'),
    Output('roc-curve', 'figure'),
    Output('comp_fig', 'figure'),
    Output('corr-graph', 'figure'),
    Output('classification-report-bar-fig', 'figure'),
    Output('churn-by-gender-pie', 'figure'),
    Output('churn-by-SeniorCitizen-sunburst', 'figure'),
    Input('model-dropdown', 'value')
)
def update_charts(model_name):
    if model_name == 'lr':
        model = lr
    elif model_name == 'rf':
        model = rf
    elif model_name == 'svc':
        model = svm
    elif model_name == 'dt':
        model = dt
    elif model_name == 'xgb':
        model = xgb
    elif model_name == 'adb':
        model = adb
    elif model_name == 'vclf':
        model = vclf

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm)
    cm.insert(0, "Actual", ["No Churn", "Churn"])
    cm.columns = ["Actual", "No Churn", "Churn"]
    cm.set_index("Actual", inplace=True)
    cm.rename_axis("Predicted", axis="columns", inplace=True)
    cm_fig = px.imshow(cm, text_auto=True)
    cm_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })


    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
            y_prob = model.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    score = metrics.auc(fpr, tpr)


    auc_fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={score:.4f})',
        labels=dict(
            x='False Positive Rate',
            y='True Positive Rate'),
        )
    auc_fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1)
    auc_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })

    prec_score = precision_score(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    report = [prec_score, acc_score, rec_score, f_score]
    report=pd.DataFrame({"Precision":prec_score, "Accuracy":acc_score, "Recall":rec_score, "F1 Score":f_score }, index=[0])
    report = report.transpose()
    report.reset_index(inplace=True)
    report.columns=["Scores",'Values']
    classification_report_bar_fig = px.bar(report, x='Scores', y="Values", title="Classification Report")
    classification_report_bar_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })

    # Create a bar chart of the evaluation metrics
    trace1 = go.Bar(x=list(models.keys()), y=accuracy, name='Accuracy')
    trace2 = go.Bar(x=list(models.keys()), y=precision, name='Precision')
    trace3 = go.Bar(x=list(models.keys()), y=recall, name='Recall')
    trace4 = go.Bar(x=list(models.keys()), y=f1, name='F1 Score')

    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(title='Model Comparison',
                    xaxis=dict(title='Model'),
                    yaxis=dict(title='Score'))

    comp_fig = go.Figure(data=data, layout=layout)
    comp_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })


    churn_counts_gender = df.groupby(['gender', 'Churn']).size().reset_index(name='count')
    churn_by_gender_fig = px.sunburst(churn_counts_gender, path=['gender', 'Churn'], values='count', color='Churn')
    churn_by_gender_fig.update_traces(textinfo='label+percent entry')
    churn_by_gender_fig.update_layout(title={
        'text': "<b>Gender-wise Distribution of Customer Churn</b>",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    })
    churn_by_gender_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })

    churn_counts_senior_citizen = df.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='count')
    churn_counts_senior_citizen['SeniorCitizen'] = churn_counts_senior_citizen['SeniorCitizen'].replace({0: 'Non-Senior', 1: 'Senior'})
    churn_by_senior_citizen_fig = px.sunburst(churn_counts_senior_citizen, path=['SeniorCitizen', 'Churn'], values='count', color='Churn')
    churn_by_senior_citizen_fig.update_traces(textinfo='label+percent entry')
    churn_by_senior_citizen_fig.update_layout(title={
        'text': "<b>Seniority-wise Distribution of Customer Churn</b>",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    })
    churn_by_senior_citizen_fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })


    crr = df_num.corr()['Churn'].sort_values(ascending=False)
    corr_graph = px.bar(x=crr.index, y=crr.values, color=crr.values, color_continuous_scale=['gray', 'blue'])

    corr_graph.update_layout(title='Correlation of Churn with other variables', xaxis_title='Features', yaxis_title='Correlation coefficient', title_font_size=18, xaxis_title_font_size=15, yaxis_title_font_size=15, height=500, width=1125)
    corr_graph.update_xaxes(tickangle=45)
    corr_graph.update_layout(
        xaxis = dict(
        tickfont = dict(size=10)))
    corr_graph.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'title_font_color':"white",
    'font_color':"white",
    'legend_title_font_color':"white",
    })
    # corr_graph.show()

    
    return cm_fig, auc_fig, comp_fig, corr_graph, classification_report_bar_fig, churn_by_gender_fig, churn_by_senior_citizen_fig
    # return cm_fig, auc_fig, comp_fig, classification_report_bar_fig

# run the app
if __name__=='__main__':
    app.run_server(debug=False)