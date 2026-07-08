import time
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

# Alpaca Trading Imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Securely load Alpaca Keys from GitHub Secrets
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

if API_KEY and SECRET_KEY:
    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
else:
    print("⚠️ Alpaca Keys missing. Running in simulation mode.")
    trading_client = None

# --- PARTICLE FILTER ENGINE ---
class ParticleFilter1D:
    def __init__(self, num_particles, initial_price):
        self.num_particles = num_particles
        # Spawn thousands of particles at the starting price
        self.particles = np.ones(num_particles) * initial_price
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, process_noise_std):
        # Move all particles randomly based on expected market volatility
        self.particles += np.random.normal(0, process_noise_std, self.num_particles)

    def update(self, measurement, measurement_noise_std):
        # Calculate how far each particle is from the actual live stock price
        distance = self.particles - measurement
        
        # Pure Numpy Gaussian Probability (Grades the particles)
        # Closer to live price = higher probability of survival
        prob = (1.0 / np.sqrt(2.0 * np.pi * measurement_noise_std**2)) * np.exp(-0.5 * (distance / measurement_noise_std)**2)
        
        self.weights *= prob
        self.weights += 1.e-300  # Avoid divide by zero errors
        self.weights /= sum(self.weights) # Normalize so all weights equal 1.0

    def estimate(self):
        # The "True Trend" is the weighted average of all surviving particles
        mean_estimate = np.average(self.particles, weights=self.weights)
        # Calculate uncertainty (standard deviation) to build our LONG/SHORT bands
        variance = np.average((self.particles - mean_estimate)**2, weights=self.weights)
        return mean_estimate, np.sqrt(variance)

    def resample(self):
        # Kill bad particles and clone the good ones
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indexes = np.searchsorted(cumulative_sum, np.random.random(self.num_particles))
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles


def run_particle_bot(ticker_symbol, deviation_threshold=1.0):
    print(f"Initializing Particle Filter (1,000 Particles) for {ticker_symbol}...")
    
    ticker = yf.Ticker(ticker_symbol)
    
    initial_df = ticker.history(period="1d", interval="1m")
    last_price = initial_df['Close'].iloc[-1] if not initial_df.empty else 1000.0
        
    # Initialize the Particle Filter
    pf = ParticleFilter1D(num_particles=1000, initial_price=last_price)
    last_processed_timestamp = None
    current_position = 0 

    print(f"Live Tracking Started. Checking prices every 15 minutes...")
    print(f"{'Time (IST)':<10} | {'Live Price':<10} | {'PF Estimate':<11} | {'Signal':<8} | {'Action'}")
    print("-" * 75)

    while True:
        try:
            df = ticker.history(period="1d", interval="1m")
            if not df.empty:
                live_price = float(df['Close'].iloc[-1])
                latest_timestamp = df.index[-1]
                
                if latest_timestamp != last_processed_timestamp:
                    last_processed_timestamp = latest_timestamp
                    
                    # 1. Predict where the trend is going
                    pf.predict(process_noise_std=0.5)
                    # 2. Update with the actual live price
                    pf.update(live_price, measurement_noise_std=2.0)
                    # 3. Get the calculated trend and uncertainty
                    pf_est, std_dev = pf.estimate()
                    # 4. Resample particles for the next loop
                    pf.resample()
                    
                    # Construct our dynamic boundaries
                    upper_band = pf_est + (deviation_threshold * std_dev)
                    lower_band = pf_est - (deviation_threshold * std_dev)
                    
                    signal = "HOLD  ⚪"
                    action_taken = "None"
                    trade_symbol = ticker_symbol.replace(".NS", "")
                    
                    # TRADE LOGIC
                    if live_price > upper_band:
                        signal = "SELL 🔴"
                        if current_position == 1: 
                            if trading_client:
                                trading_client.close_all_positions(cancel_orders=True) 
                                action_taken = "CLOSED LONG POSITION"
                            else:
                                action_taken = "SIMULATED SELL (No Keys)"
                            current_position = 0
                            
                    elif live_price < lower_band:
                        signal = "LONG  🟢"
                        if current_position <= 0:
                            if trading_client:
                                order = MarketOrderRequest(
                                    symbol=trade_symbol, 
                                    qty=1,    
                                    side=OrderSide.BUY,
                                    time_in_force=TimeInForce.GTC
                                )
                                trading_client.submit_order(order)
                                action_taken = "EXECUTED BUY"
                            else:
                                action_taken = "SIMULATED BUY (No Keys)"
                            current_position = 1
                            
                    time_str = datetime.now().strftime('%H:%M:%S')
                    print(f"{time_str:<10} | {live_price:<10.2f} | {pf_est:<11.2f} | {signal:<8} | {action_taken}")
                    
        except Exception as e:
            print(f"Error fetching data: {e}. Retrying...")
            
        time.sleep(300) 

if __name__ == "__main__":
    # We are tracking Reliance Industries on the Indian NSE
    run_particle_bot("USDEUR=X", deviation_threshold=1.4)
