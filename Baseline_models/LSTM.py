# %% [markdown]
# # LSTM + Markowitz Portfolio Optimization System
# ## Baseline Implementation for FYP
# 
# This notebook implements multiple baseline strategies for portfolio optimization:
# 1. **S&P 500 Buy-and-Hold** - Passive benchmark
# 2. **Equal-Weight Portfolio** - Simple diversification
# 3. **Traditional Markowitz** - Historical returns + optimization
# 4. **LSTM + Markowitz** - ML predictions + optimization (main baseline)
# 
# **Author:** Alasteir Ho  
# **Project:** Meta-Learning for Adaptive Multi-Source Sentiment Fusion

# %% [markdown]
# ## 1. Setup and Configuration

# %%
# Import required libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Portfolio optimization libraries
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("✓ All libraries imported successfully")

# %%
# Configuration parameters
INITIAL_CAPITAL = 10000
SEQUENCE_LENGTH = 60  # Days to look back
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

# 20 stocks from 4 different industries
STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY'],
    'Financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE']
}

ALL_STOCKS = [stock for stocks in STOCKS.values() for stock in stocks]
BENCHMARK = '^GSPC'  # S&P 500

print(f"Configuration:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"  Sequence Length: {SEQUENCE_LENGTH} days")
print(f"  Total Stocks: {len(ALL_STOCKS)}")
print(f"  LSTM Hidden Size: {HIDDEN_SIZE}")
print(f"  Training Epochs: {EPOCHS}")

# %% [markdown]
# ## 2. Model Architecture

# %%
class LSTMPredictor(nn.Module):
    """
    LSTM-based stock price predictor
    
    Architecture:
    - Input: Sequence of historical prices for all stocks
    - LSTM layers: Process temporal patterns
    - Fully connected layers: Generate price prediction
    - Output: Predicted next-day price (scaled)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Test model instantiation
test_model = LSTMPredictor(input_size=20, hidden_size=HIDDEN_SIZE, 
                          num_layers=NUM_LAYERS, dropout=DROPOUT)
print(f"✓ LSTM Model created successfully")
print(f"  Total parameters: {sum(p.numel() for p in test_model.parameters()):,}")

# %% [markdown]
# ## 3. Risk Metrics Functions

# %%
def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns
    Measures risk-adjusted return
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_sortino_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """
    Sortino Ratio = (Mean Return - Risk Free Rate) / Downside Std Dev
    Similar to Sharpe but only penalizes downside volatility
    """
    if len(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    downside_std = downside_returns.std()
    return np.sqrt(252) * excess_returns.mean() / downside_std

def calculate_max_drawdown(values):
    """
    Maximum Drawdown = Largest peak-to-trough decline
    Measures worst-case loss from peak
    """
    if len(values) == 0:
        return 0
    peak = values[0]
    max_dd = 0
    
    for value in values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def calculate_metrics(portfolio_values, benchmark_values, initial_capital):
    """Calculate comprehensive performance metrics for comparison"""
    port_values = np.array(portfolio_values)
    bench_values = np.array(benchmark_values)
    
    # Calculate returns
    port_returns = pd.Series(port_values).pct_change().dropna()
    bench_returns = pd.Series(bench_values).pct_change().dropna()
    
    # Total returns
    port_total_return = (port_values[-1] - initial_capital) / initial_capital
    bench_total_return = (bench_values[-1] - initial_capital) / initial_capital
    
    # Risk metrics
    port_sharpe = calculate_sharpe_ratio(port_returns)
    bench_sharpe = calculate_sharpe_ratio(bench_returns)
    port_sortino = calculate_sortino_ratio(port_returns)
    bench_sortino = calculate_sortino_ratio(bench_returns)
    port_max_dd = calculate_max_drawdown(port_values)
    bench_max_dd = calculate_max_drawdown(bench_values)
    
    # Volatility
    port_volatility = port_returns.std() * np.sqrt(252) if len(port_returns) > 0 else 0
    bench_volatility = bench_returns.std() * np.sqrt(252) if len(bench_returns) > 0 else 0
    
    # Win Rate
    port_win_rate = (port_returns > 0).sum() / len(port_returns) if len(port_returns) > 0 else 0
    
    return {
        'Portfolio Total Return': port_total_return,
        'Benchmark Total Return': bench_total_return,
        'Portfolio Sharpe Ratio': port_sharpe,
        'Benchmark Sharpe Ratio': bench_sharpe,
        'Portfolio Sortino Ratio': port_sortino,
        'Benchmark Sortino Ratio': bench_sortino,
        'Portfolio Max Drawdown': port_max_dd,
        'Benchmark Max Drawdown': bench_max_dd,
        'Portfolio Volatility': port_volatility,
        'Benchmark Volatility': bench_volatility,
        'Portfolio Win Rate': port_win_rate,
        'Outperformance': port_total_return - bench_total_return
    }

print("✓ Risk metrics functions defined")

# %% [markdown]
# ## 4. Data Loading and Preprocessing

# %%
def download_stock_data(tickers, start_date, end_date):
    """Download historical stock price data from Yahoo Finance"""
    print(f"Downloading data for {len(tickers)} securities...")
    data = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, 
                           progress=False, auto_adjust=True)
            
            if df.empty:
                print(f"  ✗ Warning: No data for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Handle both single-level and multi-level column indices
            if 'Close' in df.columns:
                close_prices = df['Close']
            elif isinstance(df.columns, pd.MultiIndex):
                close_prices = df['Close'][ticker] if ticker in df['Close'].columns else df['Close'].iloc[:, 0]
            else:
                close_prices = df
            
            # Ensure it's a Series
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]
            
            if len(close_prices) > 0:
                data[ticker] = close_prices
                print(f"  ✓ {ticker}: {len(close_prices)} days")
            else:
                print(f"  ✗ Warning: Empty data for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if not data:
        raise ValueError(f"No stock data was successfully downloaded!")
    
    if failed_tickers:
        print(f"\nFailed to download: {failed_tickers}")
    
    print(f"\n✓ Successfully downloaded {len(data)}/{len(tickers)} securities")
    
    # Create DataFrame with proper alignment
    result = pd.DataFrame(data)
    return result.fillna(method='ffill').fillna(method='bfill')

def create_sequences(data, seq_length):
    """Create sequences for LSTM training (sliding window approach)"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

print("✓ Data loading functions defined")

# %%
# Download data
print("=" * 70)
print("DATA COLLECTION")
print("=" * 70)

end_date = datetime.now()
start_date = end_date - timedelta(days=930)  # ~2.5 years

stock_data = download_stock_data(
    ALL_STOCKS + [BENCHMARK], 
    start_date.strftime('%Y-%m-%d'), 
    end_date.strftime('%Y-%m-%d')
)

# Remove any stocks with missing data
stock_data = stock_data.dropna(axis=1)
available_stocks = [col for col in stock_data.columns if col != BENCHMARK]

print(f"\n✓ Available stocks after cleaning: {len(available_stocks)}")
print(f"  Stocks: {', '.join(available_stocks[:10])}...")

# Split into train/test (80/20)
split_idx = int(len(stock_data) * 0.8)
train_data = stock_data[:split_idx]
test_data = stock_data[split_idx:]

print(f"\n✓ Data split:")
print(f"  Training:   {train_data.index[0].date()} to {train_data.index[-1].date()} ({len(train_data)} days)")
print(f"  Testing:    {test_data.index[0].date()} to {test_data.index[-1].date()} ({len(test_data)} days)")
print(f"  Total:      {len(stock_data)} days")

# %% [markdown]
# ## 5. Model Training

# %%
def train_model(model, train_loader, criterion, optimizer, device, epochs):
    """Train the LSTM model with early stopping"""
    model.train()
    train_losses = []
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if (epoch + 1) % 10 != 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
            print(f'  Early stopping triggered at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses

# %%
# Train separate LSTM for each stock
print("=" * 70)
print(f"TRAINING {len(available_stocks)} LSTM MODELS (ONE PER STOCK)")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

models = {}
scalers = {}

for idx, stock in enumerate(available_stocks):
    print(f"[{idx+1}/{len(available_stocks)}] Training model for {stock}...")
    
    # Prepare data for this stock
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[available_stocks])
    
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)
    
    # Find the index of current stock
    stock_idx = available_stocks.index(stock)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train[:, stock_idx])
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = LSTMPredictor(
        input_size=len(available_stocks), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train_losses = train_model(model, train_loader, criterion, optimizer, device, EPOCHS)
    
    # Store model and scaler
    models[stock] = model
    scalers[stock] = scaler
    
    print(f"  ✓ {stock} trained (Final loss: {train_losses[-1]:.6f})\n")

print(f"✓ All {len(models)} models trained successfully!")

# %% [markdown]
# ## 6. Portfolio Manager with Markowitz Optimization

# %%
class PortfolioManager:
    """
    Portfolio Manager with Markowitz Mean-Variance Optimization
    
    Features:
    - Execute buy/sell trades
    - Calculate portfolio value
    - Markowitz optimization using expected returns
    - Risk-adjusted rebalancing
    """
    
    def __init__(self, initial_capital, stocks):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.stocks = stocks
        self.holdings = {stock: 0 for stock in stocks}
        self.portfolio_values = []
        self.trades = []
        self.rebalance_dates = []
        self.optimization_metrics = []
        
    def execute_trade(self, stock, shares, price, date, action):
        """Execute a buy or sell trade"""
        if action == 'BUY':
            cost = shares * price
            if cost <= self.cash:
                self.cash -= cost
                self.holdings[stock] += shares
                self.trades.append({
                    'Date': date,
                    'Stock': stock,
                    'Action': 'BUY',
                    'Shares': shares,
                    'Price': price,
                    'Cost': cost,
                    'Cash_After': self.cash
                })
                return True
        elif action == 'SELL':
            if self.holdings[stock] >= shares:
                proceeds = shares * price
                self.cash += proceeds
                self.holdings[stock] -= shares
                self.trades.append({
                    'Date': date,
                    'Stock': stock,
                    'Action': 'SELL',
                    'Shares': shares,
                    'Price': price,
                    'Proceeds': proceeds,
                    'Cash_After': self.cash
                })
                return True
        return False
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        stock_value = sum(self.holdings[stock] * prices.get(stock, 0) 
                         for stock in self.stocks)
        return self.cash + stock_value
    
    def get_portfolio_breakdown(self, prices):
        """Get detailed portfolio breakdown"""
        breakdown = []
        total_value = 0
        
        for stock in self.stocks:
            shares = self.holdings[stock]
            if shares > 0:
                price = prices.get(stock, 0)
                value = shares * price
                total_value += value
                breakdown.append({
                    'Stock': stock,
                    'Shares': shares,
                    'Price': price,
                    'Value': value
                })
        
        # Add cash position
        breakdown.append({
            'Stock': 'CASH',
            'Shares': 1,
            'Price': self.cash,
            'Value': self.cash
        })
        total_value += self.cash
        
        # Add percentage allocations
        for item in breakdown:
            item['Allocation (%)'] = (item['Value'] / total_value * 100) if total_value > 0 else 0
        
        return breakdown, total_value
    
    def predictions_to_expected_returns(self, predictions, historical_prices):
        """Convert LSTM predictions to expected returns"""
        expected_returns_dict = {}
        
        for stock in self.stocks:
            if stock in predictions and stock in historical_prices.columns:
                current_price = historical_prices[stock].iloc[-1]
                predicted_price = predictions[stock]
                
                # Calculate expected return
                expected_return = (predicted_price - current_price) / current_price
                
                # Cap extreme predictions
                expected_return = np.clip(expected_return, -0.10, 0.10)
                
                expected_returns_dict[stock] = expected_return
            else:
                expected_returns_dict[stock] = 0.0
        
        return pd.Series(expected_returns_dict)
    
    def markowitz_optimize(self, predictions, historical_prices, current_prices, date, verbose=False):
        """Markowitz Mean-Variance Optimization"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"MARKOWITZ OPTIMIZATION: {date}")
            print(f"{'='*60}")
        
        # Convert predictions to expected returns
        mu = self.predictions_to_expected_returns(predictions, historical_prices)
        
        if verbose:
            print("\nExpected Returns (from LSTM):")
            sorted_returns = mu.sort_values(ascending=False)
            for stock, ret in sorted_returns.head(10).items():
                print(f"  {stock}: {ret:+.4f} ({ret*100:+.2f}%)")
        
        # Calculate covariance matrix
        recent_prices = historical_prices[self.stocks].tail(min(252, len(historical_prices)))
        
        try:
            S = risk_models.sample_cov(recent_prices, frequency=252)
        except:
            returns = recent_prices.pct_change().dropna()
            S = returns.cov() * 252
        
        # Run Markowitz Optimization
        try:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w >= 0)  # Long-only
            ef.add_constraint(lambda w: w <= 0.25)  # Max 25% per stock
            
            weights = ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            cleaned_weights = ef.clean_weights()
            
            expected_return, volatility, sharpe = ef.portfolio_performance(verbose=False)
            
            if verbose:
                print(f"\nOptimized Portfolio:")
                print(f"  Expected Return: {expected_return:.2%}")
                print(f"  Volatility:      {volatility:.2%}")
                print(f"  Sharpe Ratio:    {sharpe:.3f}")
                
                print(f"\nOptimal Weights (>1%):")
                for stock, weight in sorted(cleaned_weights.items(), key=lambda x: x[1], reverse=True):
                    if weight > 0.01:
                        print(f"  {stock}: {weight:.2%}")
            
            return cleaned_weights, expected_return, volatility, sharpe
            
        except Exception as e:
            if verbose:
                print(f"\n⚠️  Optimization failed: {e}")
            
            # Fallback: Equal-weight top 10
            top_stocks = mu.nlargest(10).index.tolist()
            equal_weight = 1.0 / len(top_stocks)
            cleaned_weights = {stock: (equal_weight if stock in top_stocks else 0.0) 
                             for stock in self.stocks}
            
            return cleaned_weights, None, None, None
    
    def rebalance_portfolio_markowitz(self, predictions, historical_prices, current_prices, date, verbose=False):
        """Rebalance portfolio using Markowitz optimization"""
        optimal_weights, exp_return, volatility, sharpe = self.markowitz_optimize(
            predictions, historical_prices, current_prices, date, verbose=verbose
        )
        
        portfolio_value = self.get_portfolio_value(current_prices)
        
        if verbose:
            print(f"\nPortfolio Value: ${portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"\nExecuting Trades:")
        
        # Calculate target values
        target_values = {stock: portfolio_value * weight * 0.98 
                        for stock, weight in optimal_weights.items()}
        
        current_values = {stock: self.holdings[stock] * current_prices[stock] 
                         for stock in self.stocks}
        
        trades_executed = 0
        
        # SELL positions
        for stock in self.stocks:
            current_value = current_values[stock]
            target_value = target_values.get(stock, 0)
            
            if current_value > target_value + 100:
                shares_to_sell = int((current_value - target_value) / current_prices[stock])
                if shares_to_sell > 0 and self.holdings[stock] >= shares_to_sell:
                    if self.execute_trade(stock, shares_to_sell, current_prices[stock], date, 'SELL'):
                        if verbose:
                            print(f"  SOLD  {shares_to_sell:4d} shares of {stock:6s} @ ${current_prices[stock]:7.2f}")
                        trades_executed += 1
        
        # BUY positions
        for stock in self.stocks:
            current_value = self.holdings[stock] * current_prices[stock]
            target_value = target_values.get(stock, 0)
            
            if target_value > current_value + 100:
                shares_to_buy = int((target_value - current_value) / current_prices[stock])
                cost = shares_to_buy * current_prices[stock]
                
                if shares_to_buy > 0 and cost <= self.cash:
                    if self.execute_trade(stock, shares_to_buy, current_prices[stock], date, 'BUY'):
                        if verbose:
                            print(f"  BOUGHT {shares_to_buy:4d} shares of {stock:6s} @ ${current_prices[stock]:7.2f}")
                        trades_executed += 1
        
        if verbose and trades_executed == 0:
            print("  No trades needed")
        
        self.rebalance_dates.append(date)
        
        metric = {
            'date': date,
            'portfolio_value': portfolio_value,
            'expected_return': exp_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'trades': trades_executed
        }
        self.optimization_metrics.append(metric)
        
        return metric

print("✓ Portfolio Manager class defined")

# %% [markdown]
# ## 7. Backtesting - LSTM + Markowitz (Main Baseline)

# %%
print("=" * 70)
print("BASELINE 1: LSTM + MARKOWITZ OPTIMIZATION")
print("=" * 70)

# Initialize portfolio
portfolio_lstm = PortfolioManager(INITIAL_CAPITAL, available_stocks)
portfolio_values_lstm = []
dates = []

# Set all models to eval mode
for model in models.values():
    model.eval()

rebalance_counter = 0

with torch.no_grad():
    for i in range(SEQUENCE_LENGTH, len(test_data)):
        # Get current prices
        current_date = test_data.index[i]
        current_prices = {stock: test_data[stock].iloc[i] for stock in available_stocks}
        
        # Predict next price for each stock
        predictions = {}
        for stock in available_stocks:
            test_slice = test_data[available_stocks].iloc[:i+1]
            test_scaled = scalers[stock].transform(test_slice)
            
            if len(test_scaled) >= SEQUENCE_LENGTH:
                sequence = test_scaled[-SEQUENCE_LENGTH:]
                X = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                
                pred_scaled = models[stock](X).item()
                
                # Inverse transform
                stock_idx = available_stocks.index(stock)
                dummy_array = np.zeros((1, len(available_stocks)))
                dummy_array[0, stock_idx] = pred_scaled
                pred_price = scalers[stock].inverse_transform(dummy_array)[0, stock_idx]
                
                predictions[stock] = pred_price
            else:
                predictions[stock] = current_prices[stock]
        
        # Rebalance every 5 days
        if i % 5 == 0:
            rebalance_counter += 1
            verbose = (rebalance_counter <= 2)  # Show first 2 rebalances
            
            historical_slice = test_data[available_stocks].iloc[:i+1]
            
            portfolio_lstm.rebalance_portfolio_markowitz(
                predictions, 
                historical_slice, 
                current_prices, 
                current_date,
                verbose=verbose
            )
        
        # Record portfolio value
        portfolio_value = portfolio_lstm.get_portfolio_value(current_prices)
        portfolio_values_lstm.append(portfolio_value)
        dates.append(current_date)

print(f"\n✓ LSTM + Markowitz backtest complete")
print(f"  Rebalances: {len(portfolio_lstm.rebalance_dates)}")
print(f"  Total Trades: {len(portfolio_lstm.trades)}")
print(f"  Final Value: ${portfolio_values_lstm[-1]:,.2f}")

# %% [markdown]
# ## 8. Baseline 2: S&P 500 Buy-and-Hold

# %%
print("\n" + "=" * 70)
print("BASELINE 2: S&P 500 BUY-AND-HOLD")
print("=" * 70)

# Initial S&P 500 investment
initial_spy_price = test_data[BENCHMARK].iloc[SEQUENCE_LENGTH]
spy_shares = INITIAL_CAPITAL / initial_spy_price

benchmark_values = []
for i in range(SEQUENCE_LENGTH, len(test_data)):
    benchmark_value = spy_shares * test_data[BENCHMARK].iloc[i]
    benchmark_values.append(benchmark_value)

benchmark_return = (benchmark_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"✓ S&P 500 backtest complete")
print(f"  Initial Price: ${initial_spy_price:.2f}")
print(f"  Shares: {spy_shares:.2f}")
print(f"  Final Value: ${benchmark_values[-1]:,.2f}")
print(f"  Return: {benchmark_return:.2%}")

# %% [markdown]
# ## 9. Baseline 3: Equal-Weight Portfolio

# %%
print("\n" + "=" * 70)
print("BASELINE 3: EQUAL-WEIGHT PORTFOLIO")
print("=" * 70)

# Initial equal-weight allocation
initial_prices_dict = {stock: test_data[stock].iloc[SEQUENCE_LENGTH] 
                      for stock in available_stocks}

allocation_per_stock = INITIAL_CAPITAL / len(available_stocks)
shares_equal = {stock: allocation_per_stock / price 
               for stock, price in initial_prices_dict.items()}

equal_weight_values = []
for i in range(SEQUENCE_LENGTH, len(test_data)):
    current_prices_dict = {stock: test_data[stock].iloc[i] for stock in available_stocks}
    value = sum(shares_equal[stock] * current_prices_dict[stock] 
               for stock in available_stocks)
    equal_weight_values.append(value)

equal_weight_return = (equal_weight_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"✓ Equal-weight backtest complete")
print(f"  Allocation per stock: ${allocation_per_stock:.2f}")
print(f"  Final Value: ${equal_weight_values[-1]:,.2f}")
print(f"  Return: {equal_weight_return:.2%}")

# %% [markdown]
# ## 10. Baseline 4: Traditional Markowitz (Historical Returns)

# %%
print("\n" + "=" * 70)
print("BASELINE 4: TRADITIONAL MARKOWITZ (No ML Predictions)")
print("=" * 70)

# Use historical returns instead of LSTM predictions
print("Calculating historical returns...")
historical_returns = train_data[available_stocks].pct_change().mean() * 252  # Annualized
S_traditional = risk_models.sample_cov(train_data[available_stocks])

print("\nOptimizing portfolio with historical returns...")
ef_traditional = EfficientFrontier(historical_returns, S_traditional)
ef_traditional.add_constraint(lambda w: w >= 0)
ef_traditional.add_constraint(lambda w: w <= 0.25)
traditional_weights = ef_traditional.max_sharpe(risk_free_rate=RISK_FREE_RATE)
cleaned_traditional_weights = ef_traditional.clean_weights()

print("\nTraditional Markowitz Weights (>1%):")
for stock, weight in sorted(cleaned_traditional_weights.items(), key=lambda x: x[1], reverse=True):
    if weight > 0.01:
        print(f"  {stock}: {weight:.2%}")

# Backtest traditional portfolio (buy and hold with these weights)
portfolio_traditional = PortfolioManager(INITIAL_CAPITAL, available_stocks)

# Initial allocation
print("\nInitial allocation:")
for stock, weight in cleaned_traditional_weights.items():
    if weight > 0.01:
        target_value = INITIAL_CAPITAL * weight
        shares = int(target_value / initial_prices_dict[stock])
        if shares > 0:
            portfolio_traditional.execute_trade(
                stock, shares, initial_prices_dict[stock], 
                test_data.index[SEQUENCE_LENGTH], 'BUY'
            )
            print(f"  BOUGHT {shares:4d} shares of {stock:6s} @ ${initial_prices_dict[stock]:7.2f}")

traditional_values = []
for i in range(SEQUENCE_LENGTH, len(test_data)):
    prices_dict = {stock: test_data[stock].iloc[i] for stock in available_stocks}
    value = portfolio_traditional.get_portfolio_value(prices_dict)
    traditional_values.append(value)

traditional_return = (traditional_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL

print(f"\n✓ Traditional Markowitz backtest complete")
print(f"  Final Value: ${traditional_values[-1]:,.2f}")
print(f"  Return: {traditional_return:.2%}")

# %% [markdown]
# ## 11. Comprehensive Results Comparison

# %%
print("\n" + "=" * 70)
print("COMPREHENSIVE BASELINE COMPARISON")
print("=" * 70)

# Calculate metrics for LSTM + Markowitz
metrics_lstm = calculate_metrics(portfolio_values_lstm, benchmark_values, INITIAL_CAPITAL)

print(f"\n{'RETURNS':^70}")
print(f"{'Strategy':<30} {'Final Value':>15} {'Return':>12} {'Outperf':>12}")
print(f"{'-'*70}")
print(f"{'1. S&P 500 Buy-and-Hold':<30} ${benchmark_values[-1]:>14,.2f} {benchmark_return:>11.2%} {0:>11.2%}")
print(f"{'2. Equal-Weight Portfolio':<30} ${equal_weight_values[-1]:>14,.2f} {equal_weight_return:>11.2%} {(equal_weight_return - benchmark_return):>11.2%}")
print(f"{'3. Traditional Markowitz':<30} ${traditional_values[-1]:>14,.2f} {traditional_return:>11.2%} {(traditional_return - benchmark_return):>11.2%}")
print(f"{'4. LSTM + Markowitz':<30} ${portfolio_values_lstm[-1]:>14,.2f} {metrics_lstm['Portfolio Total Return']:>11.2%} {metrics_lstm['Outperformance']:>11.2%}")

print(f"\n{'RISK-ADJUSTED PERFORMANCE':^70}")
print(f"{'Strategy':<30} {'Sharpe':>12} {'Sortino':>12} {'Max DD':>12} {'Win Rate':>12}")
print(f"{'-'*70}")

# Calculate metrics for each baseline
bench_sharpe = calculate_sharpe_ratio(pd.Series(benchmark_values).pct_change().dropna())
bench_sortino = calculate_sortino_ratio(pd.Series(benchmark_values).pct_change().dropna())
bench_maxdd = calculate_max_drawdown(benchmark_values)

eq_returns = pd.Series(equal_weight_values).pct_change().dropna()
eq_sharpe = calculate_sharpe_ratio(eq_returns)
eq_sortino = calculate_sortino_ratio(eq_returns)
eq_maxdd = calculate_max_drawdown(equal_weight_values)
eq_winrate = (eq_returns > 0).sum() / len(eq_returns)

trad_returns = pd.Series(traditional_values).pct_change().dropna()
trad_sharpe = calculate_sharpe_ratio(trad_returns)
trad_sortino = calculate_sortino_ratio(trad_returns)
trad_maxdd = calculate_max_drawdown(traditional_values)
trad_winrate = (trad_returns > 0).sum() / len(trad_returns)

print(f"{'1. S&P 500':<30} {bench_sharpe:>11.3f} {bench_sortino:>11.3f} {bench_maxdd:>11.2%} {'N/A':>12}")
print(f"{'2. Equal-Weight':<30} {eq_sharpe:>11.3f} {eq_sortino:>11.3f} {eq_maxdd:>11.2%} {eq_winrate:>11.2%}")
print(f"{'3. Traditional Markowitz':<30} {trad_sharpe:>11.3f} {trad_sortino:>11.3f} {trad_maxdd:>11.2%} {trad_winrate:>11.2%}")
print(f"{'4. LSTM + Markowitz':<30} {metrics_lstm['Portfolio Sharpe Ratio']:>11.3f} {metrics_lstm['Portfolio Sortino Ratio']:>11.3f} {metrics_lstm['Portfolio Max Drawdown']:>11.2%} {metrics_lstm['Portfolio Win Rate']:>11.2%}")

# %% [markdown]
# ## 12. Visualizations

# %%
# Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Portfolio Value Comparison
axes[0, 0].plot(dates, benchmark_values, label='S&P 500', linewidth=2, alpha=0.8)
axes[0, 0].plot(dates, equal_weight_values, label='Equal-Weight', linewidth=2, alpha=0.8)
axes[0, 0].plot(dates, traditional_values, label='Traditional Markowitz', linewidth=2, alpha=0.8)
axes[0, 0].plot(dates, portfolio_values_lstm, label='LSTM + Markowitz', linewidth=2.5, alpha=0.9)
axes[0, 0].set_title('Portfolio Value Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Portfolio Value ($)')
axes[0, 0].legend(loc='best')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# 2. Cumulative Returns
benchmark_rets = [(v - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for v in benchmark_values]
equal_rets = [(v - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for v in equal_weight_values]
trad_rets = [(v - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for v in traditional_values]
lstm_rets = [(v - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100 for v in portfolio_values_lstm]

axes[0, 1].plot(dates, benchmark_rets, label='S&P 500', linewidth=2, alpha=0.8)
axes[0, 1].plot(dates, equal_rets, label='Equal-Weight', linewidth=2, alpha=0.8)
axes[0, 1].plot(dates, trad_rets, label='Traditional Markowitz', linewidth=2, alpha=0.8)
axes[0, 1].plot(dates, lstm_rets, label='LSTM + Markowitz', linewidth=2.5, alpha=0.9)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 1].set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Return (%)')
axes[0, 1].legend(loc='best')
axes[0, 1].grid(True, alpha=0.3)

# 3. Sharpe Ratio Comparison
strategies = ['S&P 500', 'Equal-Weight', 'Traditional\nMarkowitz', 'LSTM +\nMarkowitz']
sharpe_ratios = [bench_sharpe, eq_sharpe, trad_sharpe, metrics_lstm['Portfolio Sharpe Ratio']]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

bars = axes[0, 2].bar(strategies, sharpe_ratios, color=colors, alpha=0.7)
axes[0, 2].set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Sharpe Ratio')
axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 4. Total Returns Comparison
returns = [
    benchmark_return * 100,
    equal_weight_return * 100,
    traditional_return * 100,
    metrics_lstm['Portfolio Total Return'] * 100
]

bars = axes[1, 0].bar(strategies, returns, color=colors, alpha=0.7)
axes[1, 0].set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Return (%)')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1, 0].grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 5. Max Drawdown Comparison
max_dds = [
    bench_maxdd * 100,
    eq_maxdd * 100,
    trad_maxdd * 100,
    metrics_lstm['Portfolio Max Drawdown'] * 100
]

bars = axes[1, 1].bar(strategies, max_dds, color=colors, alpha=0.7)
axes[1, 1].set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Max Drawdown (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10)

# 6. Summary Table
summary_text = f"""
BASELINE PERFORMANCE SUMMARY
{'='*50}

Initial Capital: ${INITIAL_CAPITAL:,}

1. S&P 500 Buy-and-Hold
   Final Value:  ${benchmark_values[-1]:,.2f}
   Return:       {benchmark_return:.2%}
   Sharpe:       {bench_sharpe:.3f}
   Max DD:       {bench_maxdd:.2%}

2. Equal-Weight Portfolio
   Final Value:  ${equal_weight_values[-1]:,.2f}
   Return:       {equal_weight_return:.2%}
   Sharpe:       {eq_sharpe:.3f}
   Max DD:       {eq_maxdd:.2%}

3. Traditional Markowitz
   Final Value:  ${traditional_values[-1]:,.2f}
   Return:       {traditional_return:.2%}
   Sharpe:       {trad_sharpe:.3f}
   Max DD:       {trad_maxdd:.2%}

4. LSTM + Markowitz
   Final Value:  ${portfolio_values_lstm[-1]:,.2f}
   Return:       {metrics_lstm['Portfolio Total Return']:.2%}
   Sharpe:       {metrics_lstm['Portfolio Sharpe Ratio']:.3f}
   Max DD:       {metrics_lstm['Portfolio Max Drawdown']:.2%}
   Trades:       {len(portfolio_lstm.trades)}
"""

axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("\n✓ Visualizations generated")

# %% [markdown]
# ## 13. Save Results

# %%
import os

# Create output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# Save baseline comparison
comparison_df = pd.DataFrame({
    'Strategy': [
        'S&P 500 Buy-and-Hold',
        'Equal-Weight Portfolio',
        'Traditional Markowitz',
        'LSTM + Markowitz'
    ],
    'Final Value': [
        benchmark_values[-1],
        equal_weight_values[-1],
        traditional_values[-1],
        portfolio_values_lstm[-1]
    ],
    'Return (%)': [
        benchmark_return * 100,
        equal_weight_return * 100,
        traditional_return * 100,
        metrics_lstm['Portfolio Total Return'] * 100
    ],
    'Sharpe Ratio': [
        bench_sharpe,
        eq_sharpe,
        trad_sharpe,
        metrics_lstm['Portfolio Sharpe Ratio']
    ],
    'Sortino Ratio': [
        bench_sortino,
        eq_sortino,
        trad_sortino,
        metrics_lstm['Portfolio Sortino Ratio']
    ],
    'Max Drawdown (%)': [
        bench_maxdd * 100,
        eq_maxdd * 100,
        trad_maxdd * 100,
        metrics_lstm['Portfolio Max Drawdown'] * 100
    ],
    'Win Rate (%)': [
        0,  # N/A for buy-and-hold
        eq_winrate * 100,
        trad_winrate * 100,
        metrics_lstm['Portfolio Win Rate'] * 100
    ]
})

comparison_df.to_csv(os.path.join(output_dir, 'baseline_comparison.csv'), index=False)
print(f"✓ Baseline comparison saved to {output_dir}/baseline_comparison.csv")

# Save LSTM portfolio breakdown
final_prices = {stock: test_data[stock].iloc[-1] for stock in available_stocks}
breakdown, total_value = portfolio_lstm.get_portfolio_breakdown(final_prices)
breakdown_df = pd.DataFrame(breakdown)
breakdown_df.to_csv(os.path.join(output_dir, 'lstm_portfolio_breakdown.csv'), index=False)
print(f"✓ Portfolio breakdown saved to {output_dir}/lstm_portfolio_breakdown.csv")

# Save all trades
trades_df = pd.DataFrame(portfolio_lstm.trades)
trades_df.to_csv(os.path.join(output_dir, 'lstm_all_trades.csv'), index=False)
print(f"✓ All trades saved to {output_dir}/lstm_all_trades.csv")

# Save figure
fig.savefig(os.path.join(output_dir, 'baseline_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved to {output_dir}/baseline_comparison.png")

print("\n" + "=" * 70)
print("ALL RESULTS SAVED TO ./outputs/")
print("=" * 70)




