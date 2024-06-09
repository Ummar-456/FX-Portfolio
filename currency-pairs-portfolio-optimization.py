import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import interact
import seaborn as sns
import yfinance as yf
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize
import datetime
import yfinance as yf
import pandas as pd

# List of currency pairs
currency_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'USDJPY=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
]

# Define the time period for historical data
start_date = '2020-01-01'
end_date = '2024-01-01'

# Download the data
data = yf.download(currency_pairs, start=start_date, end=end_date)['Adj Close']

# Save the data to a CSV file for backup
data.to_csv('/currency_pairs_data.csv')
# Calculate daily returns
returns = data.pct_change().dropna()

# Descriptive statistics
descriptive_stats = returns.describe()

# Skewness and kurtosis
skewness = returns.skew()
kurtosis = returns.kurtosis()

# Combine statistics into one DataFrame
stats_summary = descriptive_stats.T
stats_summary['skewness'] = skewness
stats_summary['kurtosis'] = kurtosis
stats_summary
import matplotlib.pyplot as plt

# Plot historical prices
for pair in currency_pairs:
    plt.figure(figsize=(10, 5))
    plt.plot(data[pair], label=pair)
    plt.title(f'Historical Prices for {pair}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Plot histograms of returns
for pair in currency_pairs:
    plt.figure(figsize=(10, 5))
    plt.hist(returns[pair], bins=50, edgecolor='k', alpha=0.7)
    plt.title(f'Returns Distribution for {pair}')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.show()
# Display skewness and kurtosis
skewness_kurtosis_summary = pd.DataFrame({
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

skewness_kurtosis_summary
# Define criteria for selection
skewness_threshold = 0.5
kurtosis_threshold = 3.5

# Filter based on skewness and kurtosis
optimal_pairs = skewness_kurtosis_summary[
    (skewness_kurtosis_summary['Skewness'].abs() < skewness_threshold) &
    (skewness_kurtosis_summary['Kurtosis'].abs() < kurtosis_threshold)
]
optimal_pairs
# Selected optimal pairs from the previous analysis
selected_pairs = optimal_pairs.index.tolist()

# Calculate daily returns for the selected pairs
selected_returns = returns[selected_pairs]

# Calculate mean returns and covariance matrix
mean_returns = selected_returns.mean()
cov_matrix = selected_returns.cov()


# Selected optimal pairs from the previous analysis
selected_pairs = optimal_pairs.index.tolist()

# Calculate daily returns for the selected pairs
selected_returns = returns[selected_pairs]

# Calculate mean returns and covariance matrix for daily returns
mean_returns = selected_returns.mean()
cov_matrix = selected_returns.cov()

# Number of portfolios to simulate
num_portfolios = 10000
results = np.zeros((5, num_portfolios))

# Risk-free rate for Sharpe ratio calculation (annualized, e.g., 0.045 for 4.5%)
risk_free_rate = 0.045
daily_risk_free_rate = risk_free_rate / 252  # Convert to daily

# Target return for Sortino ratio (annualized)
target_return = 0.045
daily_target_return = target_return / 252  # Convert to daily

# Variable to track minimum negative Sharpe ratio
min_negative_sharpe_ratio = float('inf')
min_negative_sharpe_portfolio = None

for i in range(num_portfolios):
    weights = np.random.random(len(selected_pairs))
    weights /= np.sum(weights)
    
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - daily_risk_free_rate) / portfolio_stddev
    
    # Calculate downside deviation
    portfolio_return_series = np.dot(selected_returns, weights)
    downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_return_series - daily_target_return)**2))
    sortino_ratio = (portfolio_return - daily_target_return) / downside_deviation if downside_deviation > 0 else np.nan
    
    results[0, i] = portfolio_return * 252  # Annualize portfolio return
    results[1, i] = portfolio_stddev * np.sqrt(252)  # Annualize portfolio volatility
    results[2, i] = sharpe_ratio
    results[3, i] = portfolio_return / portfolio_stddev  # Sharpe ratio without risk-free rate
    results[4, i] = sortino_ratio
    
    # Track the minimum negative Sharpe ratio portfolio
    if sharpe_ratio < 0 and sharpe_ratio < min_negative_sharpe_ratio:
        min_negative_sharpe_ratio = sharpe_ratio
        min_negative_sharpe_portfolio = results[:, i]

# Convert results to a DataFrame
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sharpe', 'Sortino Ratio'])

# Identify the portfolios with the minimum variance, maximum Sharpe ratio, and maximum Sortino ratio
min_variance_port = results_df.iloc[results_df['Volatility'].idxmin()]
max_sharpe_port = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]
max_sortino_port = results_df.iloc[results_df['Sortino Ratio'].idxmax()]

# Create DataFrame for the minimum negative Sharpe ratio portfolio
min_negative_sharpe_port_df = pd.DataFrame(min_negative_sharpe_portfolio, index=['Return', 'Volatility', 'Sharpe Ratio', 'Sharpe', 'Sortino Ratio']).T

# Display the results
print("Minimum Variance Portfolio:\n", min_variance_port)
print("\nMaximum Sharpe Ratio Portfolio:\n", max_sharpe_port)
print("\nMaximum Sortino Ratio Portfolio:\n", max_sortino_port)
print("\nMinimum Negative Sharpe Ratio Portfolio:\n", min_negative_sharpe_port_df)
# Plotting the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(min_variance_port[1], min_variance_port[0], color='r', marker='*', s=200, label='Minimum Variance')
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], color='b', marker='*', s=200, label='Maximum Sharpe')
plt.scatter(max_sortino_port[1], max_sortino_port[0], color='g', marker='*', s=200, label='Maximum Sortino')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return')
plt.legend()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Select the portfolio with the maximum Sortino ratio
max_sortino_index = results_df['Sortino Ratio'].idxmax()
max_sortino_portfolio = results_df.iloc[max_sortino_index]
max_sortino_weights = max_sortino_portfolio.name

# Extract the weights used for the maximum Sortino ratio portfolio
weights = np.random.random(len(selected_pairs))
weights /= np.sum(weights)
weights = weights * max_sortino_weights / np.sum(max_sortino_weights)

# Backtest the portfolio
portfolio_returns = np.dot(selected_returns, weights)
cumulative_returns = (1 + portfolio_returns).cumprod()

# Calculate backtest metrics
annualized_return = np.mean(portfolio_returns) * 252
annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
final_value = cumulative_returns[-1]

# Display metrics
# Display metrics
print("Backtest Metrics:")
print(f"Annualized Return: {annualized_return:.4f}")
print(f"Annualized Volatility: {annualized_volatility:.4f}")
print(f"Final Value: {final_value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label='Max Sortino Portfolio')
plt.title('Cumulative Returns of the Max Sortino Ratio Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
# Extract the weights of the portfolio with the maximum Sortino ratio
max_sortino_index = results_df['Sortino Ratio'].idxmax()
max_sortino_portfolio = results_df.iloc[max_sortino_index]

# Generate the weights for the max Sortino ratio portfolio
weights = np.random.random(len(selected_pairs))
weights /= np.sum(weights)
weights = weights * max_sortino_index / np.sum(max_sortino_index)

# Display the currency pairs and their respective weights
currency_weights = pd.Series(weights, index=selected_pairs)
print("Currencies and their respective weights in the Max Sortino Ratio Portfolio:")
print(currency_weights)
