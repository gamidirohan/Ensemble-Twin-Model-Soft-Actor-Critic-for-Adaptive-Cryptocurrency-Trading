#!/usr/bin/env python3
"""
Train SAC Portfolio Model using Local Parquet Data
Uses 1INCH-BTC.parquet from local Binance dataset
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn

from src.data.local_data_loader import LocalDataLoader
from src.envs.portfolio_env import PortfolioTradingEnvironment

class PortfolioCallback(BaseCallback):
    """Track portfolio performance during training"""
    def __init__(self, test_env, verbose=0):
        super().__init__(verbose)
        self.test_env = test_env
        self.best_return = -np.inf
        self.portfolio_weights = []
        
    def _on_step(self):
        if self.n_calls % 10000 == 0:
            obs, _ = self.test_env.reset()
            done = False
            episode_return = 0
            weights = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                episode_return += reward
                weights.append(action[0])
            
            if episode_return > self.best_return:
                self.best_return = episode_return
                
            self.portfolio_weights = weights
            
            print(f"\n[Step {self.n_calls}] Test Return: {episode_return:.4f} | Best: {self.best_return:.4f}")
            if weights:
                print(f"  Avg BTC Weight: {np.mean(weights)*100:.1f}%")
                
        return True


def train_local_data():
    """Train SAC model using local parquet data"""
    
    print("\n" + "="*60)
    print("SAC PORTFOLIO TRAINING - LOCAL DATA (1INCH-BTC)")
    print("="*60 + "\n")
    
    # Load local data
    print("📂 Loading local data...")
    loader = LocalDataLoader()
    df, features = loader.load_and_prepare('1INCH-BTC.parquet', limit=5000)
    
    print(f"\n✓ Loaded {len(df)} rows of data")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features shape: {features.shape}")
    
    # Split train/test
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size].copy()
    train_features = features[:train_size]
    test_data = df[train_size:].copy()
    test_features = features[train_size:]
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {len(train_data)} rows")
    print(f"  Testing:  {len(test_data)} rows")
    
    # Create environments
    print("\n🏗️  Creating environments...")
    train_env = PortfolioTradingEnvironment(train_data, train_features, initial_balance=10000)
    test_env = PortfolioTradingEnvironment(test_data, test_features, initial_balance=10000)
    
    # SAC configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=nn.ReLU
    )
    
    # Determine device (GPU if available, else CPU)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n💻 Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🤖 Creating SAC model...")
    model = SAC(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/",
        device=device
    )
    
    # Training
    print("\n🚀 Starting training...")
    print("  Total timesteps: 100,000")
    print("  Check interval: 10,000 steps\n")
    
    callback = PortfolioCallback(test_env)
    
    try:
        model.learn(
            total_timesteps=100000,
            callback=callback,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    obs, _ = test_env.reset()
    done = False
    total_reward = 0
    portfolio_weights = []
    net_worths = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        portfolio_weights.append(action[0])
        net_worths.append(test_env.net_worth)
    
    # Calculate metrics
    final_return = (test_env.net_worth - test_env.initial_balance) / test_env.initial_balance * 100
    
    # Buy and hold comparison
    buy_hold_return = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
    
    print(f"\n💰 Performance Metrics:")
    print(f"  Final Portfolio Value: ${test_env.net_worth:,.2f}")
    print(f"  Total Return:          {final_return:+.2f}%")
    print(f"  Buy & Hold Return:     {buy_hold_return:+.2f}%")
    print(f"  Outperformance:        {final_return - buy_hold_return:+.2f}%")
    
    if portfolio_weights:
        print(f"\n📈 Portfolio Statistics:")
        print(f"  Average BTC Weight: {np.mean(portfolio_weights)*100:.1f}%")
        print(f"  Max BTC Weight:     {np.max(portfolio_weights)*100:.1f}%")
        print(f"  Min BTC Weight:     {np.min(portfolio_weights)*100:.1f}%")
    
    # Save model
    model_name = f"1inch_btc_sac_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(model_name)
    print(f"\n✓ Model saved as {model_name}.zip")
    print("✓ Training complete!")
    
    return model, test_env


if __name__ == "__main__":
    model, test_env = train_local_data()
