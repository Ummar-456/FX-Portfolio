Forex Portfolio Optimization
This repository contains a Python implementation of a forex portfolio optimization strategy using Modern Portfolio Theory (MPT) and advanced risk metrics. The model selects optimal currency pairs based on statistical properties and constructs efficient portfolios through Monte Carlo simulation.

Overview
The optimization process:

Analyzes 10 major currency pairs from 2020-2025

Filters pairs based on skewness and kurtosis thresholds

Generates 10,000 random portfolios to identify the efficient frontier

Identifies optimal portfolios based on minimum variance, maximum Sharpe ratio, and maximum Sortino ratio

Backtests the optimal portfolio to evaluate performance

Results
Selected Currency Pairs
The model selected 4 currency pairs with favorable statistical properties:

EURJPY

EURUSD

GBPJPY

NZDUSD

Optimal Portfolio Weights (Maximum Sortino Ratio)
Currency Pair	Weight
EURJPY	61.80%
EURUSD	0.22%
GBPJPY	36.65%
NZDUSD	1.33%
Portfolio Performance Metrics
Metric	Value
Annualized Return	5.45%
Annualized Volatility	8.69%
Sharpe Ratio	0.1092
Maximum Drawdown	-10.86%
Final Value ($1 initial)	$1.32
Portfolio Characteristics
Three optimal portfolios were identified:

Minimum Variance Portfolio

Return: 2.38%

Volatility: 6.58%

Sharpe Ratio: -0.0203

Sortino Ratio: -0.0279

Maximum Sharpe Ratio Portfolio

Return: 5.45%

Volatility: 8.70%

Sharpe Ratio: 0.0069

Sortino Ratio: 0.0096

Maximum Sortino Ratio Portfolio

Return: 5.45%

Volatility: 8.70%

Sharpe Ratio: 0.0069

Sortino Ratio: 0.0096

Note: The Maximum Sharpe and Maximum Sortino portfolios converged to the same solution in this case.

Implementation Details
Data Source: Yahoo Finance API

Time Period: January 2020 - April 2025

Risk-Free Rate: 4.5% (annualized)

Selection Criteria:

Absolute skewness < 0.5

Absolute kurtosis < 3.5

Optimization Method: Monte Carlo simulation with 10,000 random portfolios

Visualizations
The code generates several visualizations:

Efficient frontier with highlighted optimal portfolios

Cumulative returns of the optimal portfolio

Portfolio drawdowns

Monthly returns distribution

Correlation matrix of selected currency pairs

Requirements
Python 3.7+

NumPy
Pandas
Matplotlib
Seaborn
yfinance
SciPy

Usage
python
# Clone the repository
git clone https://github.com/yourusername/forex-portfolio-optimization.git
cd forex-portfolio-optimization

# Install requirements
pip install -r requirements.txt

# Run the optimization
python forex_optimization.py
Limitations and Considerations
The model does not account for transaction costs or spreads

Past performance is not indicative of future results

Currency markets are influenced by numerous factors not captured in this model

The backtest period includes both bullish and bearish market conditions

The optimization assumes normal distribution of returns, which may not hold in extreme market conditions

License
MIT License
