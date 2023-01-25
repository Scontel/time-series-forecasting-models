import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
    y = np.sin(np.arange(len(dates)) * (2 * np.pi / 365)) * 10 + np.arange(len(dates)) * 0.05
    y += np.random.normal(0, 2, len(dates))
    return pd.DataFrame({'ds': dates, 'y': y})

def run_forecast():
    df = generate_sample_data()
    
    print("Training Prophet model...")
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    
    fig1 = model.plot(forecast)
    fig1.savefig("forecast_plot.png")
    
    fig2 = model.plot_components(forecast)
    fig2.savefig("forecast_components.png")
    
    print("Forecast complete. Plots saved.")

if __name__ == "__main__":
    run_forecast()
