### **ARIMAX Forecasting App**

This project is a **Stock Price Forecasting Application** built using **Dash** and **Plotly** for interactive data visualization. The app utilizes an **ARIMAX (AutoRegressive Integrated Moving Average with Exogenous Variables)** model to forecast stock prices based on historical data. 

The application is containerized using **Docker** and deployed on **AWS Elastic Beanstalk**. It allows users to select companies from the S&P 500, choose a time period for historical stock data, and input a forecast horizon to predict future stock prices.

#### **Features:**
- **Stock Price Forecasting**: Predicts stock prices using the ARIMAX model, with interactive visualizations for actual vs. predicted prices.
- **Data Source**: Uses `yfinance` to fetch real-time stock data.
- **Dynamic Input**: Users can select companies, time periods, and forecast horizons.
- **Accuracy Metrics**: Displays key performance metrics such as MAE, RMSE, and MAPE to evaluate model performance.
- **Dockerized**: Easily deployable in a containerized environment.
- **AWS Deployment**: Deployed on AWS Elastic Beanstalk for scalability and ease of access.

#### **Technology Stack:**
- **Python**
- **Dash & Plotly**
- **yfinance**
- **ARIMAX Model (statsmodels)**
- **Docker**
- **AWS Elastic Beanstalk**

#### **How to Run Locally:**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arimax-forecasting-app.git
   ```
2. Build and run the Docker container:
   ```bash
   docker build -t arimax-forecasting-app .
   docker run -p 8050:8050 arimax-forecasting-app
   ```
3. Open your browser and navigate to `http://localhost:8050`.

#### **Disclaimer:**
This tool is for educational purposes only and should not be used for real-life investment or trading decisions. The model is still under development and does not yet incorporate advanced techniques like market sentiment analysis or fundamental analysis.
