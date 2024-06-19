import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

# Black-Scholes formula for put option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

# Define the parameters for the option strategy
lower_range = 3200
upper_range = 3700
current_price = 3541
Date = "06/28/2024"  # Expiration date of the long put
K1 = 3500  # Strike price of long put
IV1 = 59.5  # Implied Volatility for long put
premium_paid = 110  # Premium paid for long put
num_contracts = 13.3
r = 0.01  # Risk-free rate; daily yield curve rates (use expiry length); https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield
S = np.linspace(2800, 4000, 400)  # Range of stock prices

# Calculate the time to expiration for the long put
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the long put option today
put_price_today = black_scholes_put(S, K1, T1, r, IV1 / 100)

# Calculate the payoff for the long put at T1 expiration (intrinsic value for long put)
payoff_K1_T1 = np.maximum(K1 - S, 0) - premium_paid  # Long put (intrinsic value - premium)

# Calculate the breakeven price
breakeven_price = K1 - premium_paid

# Calculate the current payoff for today
current_payoff = put_price_today - premium_paid

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_K1_T1 * num_contracts, label=f'Payoff at Expiration ({Date})', color='black')
ax.plot(S, current_payoff * num_contracts, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Long Put Strike = {K1}")
ax.axvline(breakeven_price, color='blue', linestyle='--', label=f"Breakeven Price = {breakeven_price:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table, including K1 and breakeven price
table_prices = np.linspace(3000, 4000, 19)
table_prices = np.append(table_prices, [breakeven_price, lower_range, upper_range, current_price])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_K1_T1 * num_contracts)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff * num_contracts)  # Interpolating current payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for K1 and breakeven price
for col in range(len(table_prices)):
    if table_prices[col] == breakeven_price:
        table[(0, col)].set_facecolor('#BCD7FF')
        table[(1, col)].set_facecolor('#BCD7FF')
        table[(2, col)].set_facecolor('#BCD7FF')
    elif table_prices[col] == lower_range:
        table[(0, col)].set_facecolor('#FFB6C1')
        table[(1, col)].set_facecolor('#FFB6C1')
        table[(2, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == upper_range:
        table[(0, col)].set_facecolor('#33CC33')
        table[(1, col)].set_facecolor('#33CC33')
        table[(2, col)].set_facecolor('#33CC33')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
