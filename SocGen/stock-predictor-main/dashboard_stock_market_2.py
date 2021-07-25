import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go

dataset_train = pd.read_csv('sg_train.csv')
dataset_train_copy = dataset_train.copy()
dataset_train_copy = dataset_train_copy.sort_values(by='Date')
dataset_test = pd.read_csv('sg_test_predict.csv')
dataset_test = dataset_test.drop(columns=['Unnamed: 0'])

fig = go.Figure()
fig.add_trace(go.Candlestick(x=pd.to_datetime(dataset_train_copy['Date']),open=dataset_train_copy['Open'],high=dataset_train_copy['High'],
                             low=dataset_train_copy['Low'],close=dataset_train_copy['Close'],increasing=dict(line=dict(color='blue')),
                             decreasing=dict(line=dict(color='red')),name='Real'))
fig.add_trace(go.Candlestick(x=pd.to_datetime(dataset_test['Date']),open=dataset_test['Open'],high=dataset_test['High'],
                             low=dataset_test['Low'],close=dataset_test['Close'],increasing=dict(line=dict(color='green')),
                             decreasing=dict(line=dict(color='pink')),name='Predicted'))
fig.update_layout(title='Stock Price Trend',title_x=0.5,xaxis=dict(tickformat='%Y-%m-%d',title='Date', nticks=10, tickangle=-45), yaxis_title='Stock Price')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
server = app.server
app.layout = html.Div(
    [
        dbc.Row([
            dbc.Col((html.H1('Stock Predictor',
                                     style={'textAlign': 'center', 'color': 'white', 'marginTop': 90})), width=12)
        ], style={'background-color': '#87D3F8', 'marginBottom': 20, 'height': 200}),
        html.Div([
                    dbc.Row([
                        dbc.Col(html.H2(html.B('Predictions for Open,High,Low and Close Prices'),
                                        style={'textAlign': 'left', 'marginBottom': 30, 'marginLeft': 10}), width=12)])
                    ]),
        html.Div([
                    dbc.Row([
                        dbc.Col(html.H5('Select the Range of dates using the Range slider below the graph',style={'textAlign':'left','marginBottom':20,'marginLeft':10}),width=12)])
                    ]),
        dbc.Row([
                    dbc.Col(dcc.Graph(id='candle-stick-chart', figure=fig, config={'displayModeBar': False})),
                ]),
            ])

if __name__ == '__main__':
    app.run_server(debug=False)

