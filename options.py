import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime

# Black-Scholes formula for call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

# Define the parameters for the option strategy
S = np.linspace(3000, 4000, 400)  # Range of stock prices
r = 0.01  # Risk-free rate
Date = "06/21/2024"  # Expiration date of short call
K1 = 3450  # Strike price of short call
IV1 = 50.8  # Implied Volatility for short call
premium_received = 115.6  # Premium received for short call

# Calculate the time to expiration for the short call
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the short call option today
call_price_today = black_scholes_call(S, K1, T1, r, IV1 / 100)

# Calculate the payoff for the short call at T1 expiration (intrinsic value for short call)
payoff_K1_T1 = -np.maximum(S - K1, 0) + premium_received  # Short call (intrinsic value + premium)

# Calculate the breakeven price
breakeven_price = K1 + premium_received

# Calculate the current payoff for today
current_payoff = -call_price_today + premium_received

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_K1_T1, label=f'Payoff at Expiration of Short Call ({int(T1*365)} days)', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Short Call Strike = {K1}")
ax.axvline(breakeven_price, color='blue', linestyle='--', label=f"Breakeven Price = {breakeven_price:.2f}")
ax.legend(fontsize=10)
ax.grid(True)

# Selecting specific prices for the table, including K1 and breakeven price
table_prices = np.linspace(3000, 4000, 17)
table_prices = np.append(table_prices, [K1, breakeven_price])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_K1_T1)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices

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
    if table_prices[col] == K1:
        table[(0, col)].set_facecolor('#FFB6C1')
        table[(1, col)].set_facecolor('#FFB6C1')
        table[(2, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == breakeven_price:
        table[(0, col)].set_facecolor('#BCD7FF')
        table[(1, col)].set_facecolor('#BCD7FF')
        table[(2, col)].set_facecolor('#BCD7FF')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()