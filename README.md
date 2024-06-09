# Currency Pairs Portfolio Optimization

This project focuses on optimizing a portfolio of currency pairs using historical data. The optimization criteria include minimizing variance, maximizing the Sharpe ratio, maximizing the Sortino ratio, and minimizing negative Sharpe ratios. The project also involves backtesting the optimal portfolios to evaluate their performance.

## Table of Contents
- [Introduction](#introduction)
- [Data Extraction](#data-extraction)
- [Statistical Analysis and EDA](#statistical-analysis-and-eda)
- [Portfolio Optimization](#portfolio-optimization)
- [Backtesting](#backtesting)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the application of Modern Portfolio Theory (MPT) to optimize a portfolio of currency pairs. The optimization process considers multiple financial metrics to find the best asset allocation. The results are backtested to evaluate the historical performance of the optimized portfolios.

## Data Extraction

Historical data for ten currency pairs is extracted from Yahoo Finance. The selected pairs are:
- EURUSD=X
- GBPUSD=X
- AUDUSD=X
- NZDUSD=X
- USDCAD=X
- USDCHF=X
- USDJPY=X
- EURGBP=X
- EURJPY=X
- GBPJPY=X

## Statistical Analysis and EDA

Perform descriptive statistical analysis and exploratory data analysis (EDA) on the extracted data. The key metrics include mean returns, standard deviations, skewness, and kurtosis. This helps in selecting the most promising pairs for portfolio optimization.

## Portfolio Optimization

Simulate 10,000 random portfolios and calculate:
- Expected annualized return
- Annualized volatility
- Sharpe ratio (annual risk-free rate: 4.5%)
- Sortino ratio (annual target return: 4.5%)

Identify the following optimal portfolios:
1. Minimum Variance Portfolio
2. Maximum Sharpe Ratio Portfolio
3. Maximum Sortino Ratio Portfolio
4. Minimum Negative Sharpe Ratio Portfolio

## Backtesting

Backtest the portfolio with the maximum Sortino ratio to evaluate its historical performance. Key metrics include:
- Annualized Return: 1.05%
- Annualized Volatility: 5.76%
- Final Value: 1.0370

## Results

- **Maximum Sortino Ratio Portfolio Weights:**
  - AUDUSD=X: 4.40%
  - EURJPY=X: 17.29%
  - EURUSD=X: 20.16%
  - GBPJPY=X: 24.88%
  - NZDUSD=X: 27.66%
  - USDCHF=X: 5.60%

- **Backtest Metrics:**
  - Annualized Return: 1.05%
  - Annualized Volatility: 5.76%
  - Final Value: 1.0370

## Conclusion

The project successfully demonstrated the application of MPT to currency pairs, identifying stable and moderately profitable investment strategies. Future work could extend the analysis to more currency pairs, different risk-free rates, or other asset classes.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/currency-pairs-portfolio-optimization.git
    cd currency-pairs-portfolio-optimization
    ```
2. Install the required dependencies.
3. Run the Python scripts to perform data extraction, statistical analysis, portfolio optimization, and backtesting.

## Dependencies

- numpy
- pandas
- matplotlib
- yfinance

Install the required packages using:
```bash
pip install numpy pandas matplotlib yfinance
Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.
This project is licensed under the MIT License.
