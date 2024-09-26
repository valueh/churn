import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from pickle import load
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import xgboost

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])
server = app.server

city_options = ['Москва', 'Новосибирск', 'Владивосток']
gender_options = ['Мужчина', 'Женщина']
slider_marks = {0: {'label': '0%'}, 20: {'label': '20%'}, 40: {'label': '40%'}, 60: {'label': '60%'}, 80: {'label': '80%'}, 100: {'label': '100%'}}
preds_df = pd.read_csv('data/preds.csv')
with open('data/model.pkl', 'rb') as f:
    clf = load(f)

def create_scores_fig():
    scores = pd.Series(data=clf.get_booster().get_score(importance_type='gain'), name='score').sort_values().round().astype('int')

    fig = px.bar(
        data_frame=scores,
        y='score',
        labels={'score': 'Score', 'index': 'Feature'},
        template='plotly_white',
    ).update_layout(
        margin=dict(l=75, t=50, r=50, b=50),
    )

    return fig

def create_roc_fig(): 
    fpr, tpr, thresholds = roc_curve(preds_df.y_true, preds_df.y_pred)
    roc_df = pd.DataFrame({'threshold': thresholds, 'fpr': fpr, 'tpr': tpr})
    roc_df.threshold = roc_df.threshold.round(4)
    roc_df = roc_df.groupby('threshold', as_index=False).mean()
    roc_df = roc_df[roc_df.threshold < 1]

    fig = px.line(
        data_frame=roc_df,
        x=roc_df.fpr,
        y=roc_df.tpr,
        hover_data={'threshold': ':.1%', 'fpr': ':.1%', 'tpr': ':.1%'},
        labels={'threshold': 'Threshold', 'fpr': 'FPR', 'tpr': 'TPR'},
        template='plotly_white',
    ).update_layout(
        margin=dict(l=75, t=50, r=50, b=50),
        xaxis_tickformat='.1%',
        yaxis_tickformat='.1%',
    )

    return fig

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H4(['Классификация банковских клиентов'], className='text-center'),
        ]),
    ], class_name='mt-2 mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(['Расчет вероятности'], class_name='text-center'),
                dbc.CardBody([
                    dbc.Col([
                        html.Div(['City']),
                        dcc.Dropdown(options=city_options, value='Москва', clearable=False, className='mb-2', id='dropdown-city'),
                        html.Div(['Gender']),
                        dcc.Dropdown(options=gender_options, value='Мужчина', clearable=False, className='mb-2', id='dropdown-gender'),
                        html.Div(['Age']),
                        dcc.Input(type='number', min=18, step=1, value=30, required=True, className='mb-2 w-100', id='input-age'),
                        html.Div(['Balance']),
                        dcc.Input(type='number', min=0, step=1, value=500000, required=True, className='mb-2 w-100', id='input-balance'),
                        html.Div(['NumOfProducts']),
                        dcc.Input(type='number', min=1, step=1, value=2, required=True, className='mb-2 w-100', id='input-products'),
                        html.Div(['IsActiveUser']),
                        dcc.Dropdown(options=[{'label': 'Да', 'value': 1}, {'label': 'Нет', 'value': 0}], value=1, clearable=False, className='mb-2', id='dropdown-active'),
                        html.Button(['PREDICT'], className='mb-2 w-100', id='btn-predict'),
                        html.Div(id='div-predict'),
                    ], class_name='d-flex flex-column justify-content-between'),
                ], class_name='d-flex flex-column'),
            ], class_name='h-100'),
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(['Важность признаков'], class_name='text-center'),
                dbc.CardBody([
                    dcc.Graph(figure=create_scores_fig(), className='flex-grow-1'),
                ], class_name='d-flex flex-column'),
            ], class_name='h-100'),
        ], md=8),
    ], class_name='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(['Кривая ROC'], class_name='text-center'),
                dbc.CardBody([
                    html.Div([f'AUC={roc_auc_score(y_true=preds_df.y_true, y_score=preds_df.y_pred):.2f}'], className='mb-2 text-center'),
                    dcc.Graph(figure=create_roc_fig(), className='flex-grow-1'),
                ], class_name='d-flex flex-column'),
            ], class_name='h-100'),
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(['Матрица ошибок'], class_name='text-center'),
                dbc.CardBody([
                    html.Div(id='div-roc', className='mb-2 text-center'),
                    dcc.Graph(figure={}, className='mb-2 flex-grow-1', id='fig-matrix'),
                    dcc.Slider(min=0, max=100, step=0.1, marks=slider_marks, value=50, tooltip={'placement': 'bottom', 'always_visible': False, 'template': 'Threshold={value}%'}, id='slider-threshold'),
                ], class_name='d-flex flex-column'),
            ], class_name='h-100'),
        ], md=4),
    ], class_name='mb-4'),
], fluid=True, class_name='dbc')

@callback(
    Output('div-predict', 'children'),
    State('dropdown-city', 'value'),
    State('dropdown-gender', 'value'),
    State('input-age', 'value'),
    State('input-balance', 'value'),
    State('input-products', 'value'),
    State('dropdown-active', 'value'),
    Input('btn-predict', 'n_clicks'),
)
def predict_input(city, gender, age, balance, products, active, n):
    user_df = pd.DataFrame({
        'NumOfProducts': [products],
        'IsActiveUser': [bool(active)],
        'Age': [age],
        'Gender': pd.Categorical([gender], gender_options),
        'City': pd.Categorical([city], city_options),
        'Balance': [balance],
    })
    if user_df.isna().sum().sum() == 0:
        proba = clf.predict_proba(user_df)[0, 1]
        return f'Вероятность ухода клиента: {proba:.1%}'
    else:
        return 'Некорректные данные'

@callback(
    Output('div-roc', 'children'),
    Output('fig-matrix', 'figure'),
    Input('slider-threshold', 'value'),
)
def update_matrix_fig(threshold):
    threshold = threshold / 100
    cm = confusion_matrix(preds_df.y_true, preds_df.y_pred > threshold)
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    metrics = f'FPR={fpr:.1%}, TPR={tpr:.1%}'

    fig = px.imshow(
        img=cm,
        x=['False', 'True'],
        y=['False', 'True'],
        labels={'x': 'Predicted label', 'y': 'Real label'},
        color_continuous_scale='Plotly3',
        template='plotly_white',
    ).update_traces(
        texttemplate='%{z:.0f}',
        hovertemplate='Predicted label=%{x}<br>Real label=%{y}<br>Count=%{z:.0f}<extra></extra>',
    ).update_layout(
        margin=dict(l=75, t=50, r=50, b=50),
        coloraxis_showscale=False,
    )

    return metrics, fig

if __name__ == '__main__':
    app.run(debug=True, dev_tools_ui=False)
