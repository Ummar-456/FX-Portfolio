import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import seaborn as sns

# List of currency pairs
currency_pairs = [
    'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'USDCAD=X',
    'USDCHF=X', 'USDJPY=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
]

# Define the time period for historical data
start_date = '2020-01-01'
end_date = '2025-04-25'

# Download the data
data = yf.download(currency_pairs, start=start_date, end=end_date)['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate descriptive statistics
descriptive_stats = returns.describe()
skewness = returns.skew()
kurtosis = returns.kurtosis()

# Combine statistics into one DataFrame
stats_summary = descriptive_stats.T
stats_summary['skewness'] = skewness
stats_summary['kurtosis'] = kurtosis

# Display skewness and kurtosis
skewness_kurtosis_summary = pd.DataFrame({
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

# Define criteria for selection
skewness_threshold = 0.5
kurtosis_threshold = 3.5

# Filter based on skewness and kurtosis
optimal_pairs = skewness_kurtosis_summary[
    (skewness_kurtosis_summary['Skewness'].abs() < skewness_threshold) &
    (skewness_kurtosis_summary['Kurtosis'].abs() < kurtosis_threshold)
]

# Selected optimal pairs from the previous analysis
selected_pairs = optimal_pairs.index.tolist()

# Calculate daily returns for the selected pairs
selected_returns = returns[selected_pairs]

# Calculate mean returns and covariance matrix
mean_returns = selected_returns.mean()
cov_matrix = selected_returns.cov()

# Number of portfolios to simulate
num_portfolios = 10000

# Risk-free rate for Sharpe ratio calculation (annualized, e.g., 0.045 for 4.5%)
risk_free_rate = 0.045
daily_risk_free_rate = risk_free_rate / 252  # Convert to daily

# Target return for Sortino ratio (annualized)
target_return = 0.045
daily_target_return = target_return / 252  # Convert to daily

# Arrays to store results and weights
results = np.zeros((5, num_portfolios))
all_weights = np.zeros((num_portfolios, len(selected_pairs)))

# Monte Carlo simulation
for i in range(num_portfolios):
    weights = np.random.random(len(selected_pairs))
    weights /= np.sum(weights)
    
    # Store the weights
    all_weights[i,:] = weights
    
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

# Convert results to a DataFrame
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sharpe', 'Sortino Ratio'])

# Identify the portfolios with the minimum variance, maximum Sharpe ratio, and maximum Sortino ratio
min_variance_idx = results_df['Volatility'].idxmin()
max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
max_sortino_idx = results_df['Sortino Ratio'].idxmax()

min_variance_port = results_df.iloc[min_variance_idx]
max_sharpe_port = results_df.iloc[max_sharpe_idx]
max_sortino_port = results_df.iloc[max_sortino_idx]

# Get the weights for each optimal portfolio
min_variance_weights = all_weights[min_variance_idx, :]
max_sharpe_weights = all_weights[max_sharpe_idx, :]
max_sortino_weights = all_weights[max_sortino_idx, :]

# Display the results
print("Minimum Variance Portfolio:\n", min_variance_port)
print("\nMaximum Sharpe Ratio Portfolio:\n", max_sharpe_port)
print("\nMaximum Sortino Ratio Portfolio:\n", max_sortino_port)

# Plotting the efficient frontier
plt.figure(figsize=(10, 7))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(min_variance_port['Volatility'], min_variance_port['Return'], color='r', marker='*', s=200, label='Minimum Variance')
plt.scatter(max_sharpe_port['Volatility'], max_sharpe_port['Return'], color='b', marker='*', s=200, label='Maximum Sharpe')
plt.scatter(max_sortino_port['Volatility'], max_sortino_port['Return'], color='g', marker='*', s=200, label='Maximum Sortino')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return')
plt.legend()
plt.show()

# Display the currency pairs and their respective weights for the Max Sortino portfolio
max_sortino_currency_weights = pd.Series(max_sortino_weights, index=selected_pairs)
print("\nCurrencies and their respective weights in the Max Sortino Ratio Portfolio:")
print(max_sortino_currency_weights)

# Backtest the Max Sortino portfolio
portfolio_returns = np.dot(selected_returns, max_sortino_weights)
cumulative_returns = (1 + portfolio_returns).cumprod()

# Calculate backtest metrics
annualized_return = np.mean(portfolio_returns) * 252
annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
final_value = cumulative_returns[-1]


# Calculate drawdowns
cumulative_returns_series = pd.Series(cumulative_returns)
previous_peaks = cumulative_returns_series.cummax()
drawdowns = (cumulative_returns_series - previous_peaks) / previous_peaks
# Display backtest metrics
print("\nBacktest Metrics:")
print(f"Annualized Return: {annualized_return:.4f}")
print(f"Annualized Volatility: {annualized_volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
print(f"Final Value (starting with $1): ${final_value:.4f}")

# Plot the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns, label='Max Sortino Portfolio')
plt.title('Cumulative Returns of the Max Sortino Ratio Portfolio')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()



# Plot drawdowns
plt.figure(figsize=(12, 6))
plt.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3)
plt.plot(drawdowns, color='red', label='Drawdowns')
plt.title('Portfolio Drawdowns')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()


# Calculate monthly returns
portfolio_returns_series = pd.Series(portfolio_returns, index=selected_returns.index)

# Now you can resample to monthly frequency
monthly_returns = portfolio_returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)

# Plot monthly returns
plt.figure(figsize=(12, 6))
monthly_returns.plot(kind='bar', color=np.where(monthly_returns > 0, 'green', 'red'))
plt.title('Monthly Returns')
plt.xlabel('Month')
plt.ylabel('Return')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate correlation matrix of selected pairs
correlation_matrix = selected_returns.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Selected Currency Pairs')
plt.show()
