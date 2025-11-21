# Ensemble Twin-Model Soft Actor-Critic for Adaptive Cryptocurrency Trading

A reinforcement learning-based cryptocurrency trading system using Soft Actor-Critic (SAC) agents in a twin-model ensemble for adaptive portfolio management. The system fetches real-time data, engineers features with technical indicators, trains models with hyperparameter optimization, and supports backtesting and live trading.

## Features
- **Twin-Model Ensemble**: Two SAC agents that compete and switch based on performance for adaptability.
- **Portfolio Optimization**: Continuous action spaces for dynamic BTC/cash allocation.
- **Hyperparameter Tuning**: Optuna and Ray-based optimization for model improvement.
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more for feature engineering.
- **Backtesting & Analysis**: Comprehensive metrics including Sharpe ratio, drawdown, and visualizations.
- **Live Trading**: Real-time execution with the twin system.
- **Modular Architecture**: Organized into data, environments, models, backtesting, optimization, and analysis.

## Project Structure

### Folders
- **`src/data/`**: Data fetching and preprocessing scripts.
  - `data_fetcher.py`: Fetches OHLCV data from Binance, adds technical indicators, and prepares features.
  - Other fetchers for different periods or exchanges (e.g., Kraken).
- **`src/envs/`**: Reinforcement learning environments.
  - `trading_env.py`: Discrete action trading environment (buy/hold/sell).
  - `portfolio_env.py`: Continuous action portfolio environment for SAC.
  - Variants like `trading_env_fixed.py` for fixed strategies.
- **`src/models/`**: Trained models and systems.
  - `twin_system.py`: Core twin-model SAC system with switching logic.
  - `live_twin_trader.py`: Live trading implementation.
- **`src/backtesting/`**: Backtesting tools.
  - `backtest.py`: Simulates trades and calculates metrics.
  - Variants for different scenarios (e.g., short periods, multi-asset).
- **`src/optimization/`**: Hyperparameter optimization.
  - `hyperopt.py`: Ray-based tuning.
  - `hyperopt_optuna.py`: Optuna-based optimization.
- **`src/analysis/`**: Performance analysis and debugging.
  - `analyze_performance.py`: Detailed metrics and charts.
  - `debug_portfolio.py`: Debugging tools.
- **`tests/`**: Unit and integration tests.
  - Scripts to validate data, environments, and models.
- **`scripts/`**: Execution scripts for training, backtesting, etc.
  - `train_portfolio.py`: Trains the twin SAC system.
  - `run_backtest.py`: Runs backtests.
  - `run_optuna_hyperopt.py`: Optimizes hyperparameters.

### Key Files
- `main.py`: Basic entry point for PPO training/backtesting (legacy).
- `requirements.txt`: Dependencies (use with `uv` and `myenv`).
- `.gitignore`: Ignores virtual envs, models, and sensitive files.

## Setup

### Prerequisites
- Python 3.12+
- `uv` package manager (for virtual environments).

### Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/gamidirohan/Ensemble-Twin-Model-Soft-Actor-Critic-for-Adaptive-Cryptocurrency-Trading.git
   cd Ensemble-Twin-Model-Soft-Actor-Critic-for-Adaptive-Cryptocurrency-Trading
   ```

2. Create and activate virtual environment:
   ```bash
   uv venv myenv
   uv pip install -r requirements.txt --python myenv
   # Or activate: myenv\Scripts\activate (Windows)
   ```

3. Verify setup:
   ```bash
   uv run --python myenv python -c "import src.data.data_fetcher; print('Setup OK')"
   ```

## Usage

### 1. Data Fetching
Fetch and preprocess cryptocurrency data with technical indicators.

**Command**:
```bash
uv run --python myenv python src/data/data_fetcher.py
# Or for specific scripts: uv run --python myenv python scripts/fetch_binanceus_data.py
```

**What it does**:
- Connects to Binance API (or Kraken) via CCXT.
- Downloads OHLCV data (e.g., BTC/USDT, 1h timeframe).
- Adds indicators: RSI, MACD, Bollinger Bands, returns, log returns.
- Prepares features for RL (normalizes prices, handles NaNs).
- Outputs: Pandas DataFrame with features ready for training.

**Output**: Data saved in memory or files; check console for sample data.

### 2. Feature Engineering/Cleaning
Data cleaning and formatting happens automatically in `data_fetcher.py`:
- Handles missing data with dropna().
- Normalizes features (e.g., relative to first close price).
- Splits into train/test sets (80/20 by default).

**Manual Check**:
```bash
uv run --python myenv python -c "from src.data.data_fetcher import BinanceDataFetcher; df = BinanceDataFetcher().fetch_ohlcv(); print(df.head())"
```

### 3. Training
Train the SAC models (single or twin system).

**Basic Training** (Single SAC):
```bash
uv run --python myenv python scripts/train_portfolio.py
```

**Twin System Training**:
- Uses `src/models/twin_system.py`.
- Trains two SAC agents in parallel.
- Includes retraining every 24h and hyperopt every 168h.

**Parameters**: Edit `train_portfolio.py` for epochs, learning rate, etc.

**Output**: Saves models (e.g., `sac_model.zip`) and logs performance.

### 4. Executing/Backtesting
Simulate trades on historical data.

**Command**:
```bash
uv run --python myenv python scripts/run_backtest.py
# Or comprehensive: uv run --python myenv python scripts/run_comprehensive_backtest.py
```

**What it does**:
- Loads trained model.
- Runs simulation in environment.
- Calculates metrics: returns, Sharpe, drawdown, win rate.
- Generates plots (e.g., portfolio value over time).

**Output**: Console metrics + `backtest_results.png`.

### 5. Hyperparameter Optimization
Tune model parameters for better performance.

**Command**:
```bash
uv run --python myenv python scripts/run_optuna_hyperopt.py
# Or distributed: uv run --python myenv python scripts/run_full_hyperopt.py
```

**What it does**:
- Uses Optuna/Ray to search hyperparameters (e.g., learning rate, network size).
- Runs trials, evaluates on validation set.
- Saves best model.

**Time**: 10-60 minutes; monitor progress in console.

### 6. Live Trading
Deploy the twin system for real-time trading (use with caution).

**Command**:
```bash
uv run --python myenv python src/models/live_twin_trader.py
```

**What it does**:
- Fetches live data.
- Switches between twin models based on performance.
- Executes trades (integrate with exchange API).

**Note**: Requires API keys; test on paper trading first.

### 7. Analysis and UI Viewing
Analyze performance and visualize results.

**Performance Analysis**:
```bash
uv run --python myenv python src/analysis/analyze_performance.py
```

**Debugging**:
```bash
uv run --python myenv python src/analysis/debug_portfolio.py
```

**Viewing**:
- Metrics: Sharpe ratio, total return, drawdown.
- Plots: Matplotlib charts saved as PNG.
- Logs: Check console or `logs/` folder.

## Metrics Tracked
- **Total Return**: Portfolio growth vs. buy-and-hold.
- **Sharpe Ratio**: Risk-adjusted returns.
- **Maximum Drawdown**: Largest peak-to-trough decline.
- **Win Rate**: Percentage of profitable trades.
- **Number of Trades**: Trading frequency.
- **Model Switching**: Twin system performance logs.

## Contributing
- Use `uv` and `myenv` for all Python operations.
- Follow modular structure; update imports if adding files.
- Test changes with `tests/` scripts.

## License
MIT License.