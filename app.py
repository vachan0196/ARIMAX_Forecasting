import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Fetch S&P 500 tickers
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)
    df = sp500_table[0]
    sp500_tickers = dict(zip(df['Security'], df['Symbol']))
    return sp500_tickers

sp500_tickers = get_sp500_tickers()
company_options = [{'label': company, 'value': ticker} for company, ticker in sp500_tickers.items()]

# Mapping for periods
period_mapping = {
    '6 months': '6mo',
    '1 year': '1y',
    '5 years': '5y',
    'All (entire history)': 'max'
}

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Expose the server for deployment

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stock Price Prediction - ARIMAX Model"), className="text-center mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.P(
            "⚠️ Warning: This tool is for educational purposes only. The model is incomplete and does not incorporate "
            "advanced techniques such as market sentiment analysis or fundamental analysis. Do not use these predictions "
            "for real-life trading or investment decisions.", 
            style={'color': 'red', 'fontWeight': 'bold'}
        ), className="text-center mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Company:"),
            dcc.Dropdown(
                id='company-dropdown',
                options=company_options,
                value='AAPL',  # Set a default value
                multi=False
            ),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Time Period:"),
            dcc.Dropdown(
                id='period-dropdown',
                options=[{'label': period, 'value': period_mapping[period]} for period in period_mapping],
                value='1y',
                multi=False
            ),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Enter Forecast Horizon (days):"),
            dcc.Input(id='horizon-input', type='number', value=5, min=1)
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Predict", id='predict-button', color='primary', className='w-100')
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-indicator",
                type="default",
                children=html.Div(id='output-results', className="mt-4")
            )
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='price-graph')
        ], width=12),
    ]),
], fluid=True)

# Function to run ARIMAX predictions
def run_arimax(stock_data, horizon):
    endog = stock_data['Close']
    exog = stock_data[['Volume']]
    
    order = (2, 1, 2)
    
    train_size = int(len(endog) * 0.8)
    train_endog = endog[:train_size]
    train_exog = exog[:train_size]
    test_endog = endog[train_size:]
    test_exog = exog[train_size:]

    predictions = []
    forecast_blocks = []

    # Rolling window forecast
    for i in range(len(test_endog) - horizon + 1):
        y_train = endog[:train_size + i]
        X_train = exog[:train_size + i]
        
        X_test = exog[train_size + i: train_size + i + horizon]
        y_test = endog[train_size + i: train_size + i + horizon]
        
        model = sm.tsa.statespace.SARIMAX(y_train, exog=X_train, order=order)
        model_fit = model.fit(disp=False)
        
        forecast = model_fit.forecast(steps=horizon, exog=X_test)
        forecast_blocks.append((y_test.index, forecast.values))
        predictions.extend(forecast.values)
    
    return predictions[-horizon:], test_endog[-horizon:]

# Callback to run ARIMAX, display predictions, and update graph with accuracy
@app.callback(
    [Output('output-results', 'children'), Output('price-graph', 'figure')],
    Input('predict-button', 'n_clicks'),
    [State('company-dropdown', 'value'), State('period-dropdown', 'value'), State('horizon-input', 'value')]
)
def get_predictions(n_clicks, ticker, period, horizon):
    if n_clicks is None:
        return dash.no_update, dash.no_update

    # Download stock data based on ticker
    stock_data = yf.download(ticker, period=period)

    # Clean the stock data
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.dropna(inplace=True)

    # Run ARIMAX model
    predicted_prices, actual_prices = run_arimax(stock_data, horizon)

    # Calculate accuracy metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    accuracy = 100 - mape

    # Display predicted prices and accuracy
    result_text = (
        f"Predicted prices for the next {horizon} days:\n" +
        "\n".join([f"{p:.2f}" for p in predicted_prices]) +
        f"\n\nMean Absolute Error (MAE): {mae:.4f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
        f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n"
        f"Accuracy: {accuracy:.2f}%"
    )
    
    # Create graph of actual vs predicted stock prices
    fig = go.Figure()

    # Plot actual prices
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Prices'))

    # Add predicted prices at the end of the time series
    prediction_index = pd.date_range(start=stock_data.index[-1], periods=horizon+1, freq='D')[1:]
    fig.add_trace(go.Scatter(x=prediction_index, y=predicted_prices, mode='lines', name='Predicted Prices', line=dict(dash='dash')))

    fig.update_layout(
        title=f'{ticker} Stock Prices and Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    return html.Pre(result_text), fig

# Run the app locally for testing
if __name__ == '__main__':
    app.run_server(debug=True)
