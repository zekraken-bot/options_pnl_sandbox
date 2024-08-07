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
lower_range = 2000
upper_range = 4000
lower_choice = 2300
upper_choice = 2800
Date = "08/16/2024"  # Expiration date of the long call
K1 = 2600  # Strike price of long call
IV1 = 82.5 # Implied Volatility for long call
premium_paid = 115.7 # Premium paid for long call
num_contracts = 1.5
r = 0.01  # Risk-free rate; daily yield curve rates (use expiry length); https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the long call
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the long call option today
call_price_today = black_scholes_call(S, K1, T1, r, IV1 / 100)

# Calculate the payoff for the long call at T1 expiration (intrinsic value for long call)
payoff_K1_T1 = np.maximum(S - K1, 0) - premium_paid  # Long call (intrinsic value - premium)

# Calculate the breakeven price
breakeven_price = K1 + premium_paid

# Calculate the current payoff for today
current_payoff = call_price_today - premium_paid

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_K1_T1 * num_contracts, label=f'Payoff at Expiration ({Date})', color='black')
ax.plot(S, current_payoff * num_contracts, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Long Call Strike = {K1}")
ax.axvline(breakeven_price, color='blue', linestyle='--', label=f"Breakeven Price = {breakeven_price:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table, including K1 and breakeven price
table_prices = np.linspace(lower_range, upper_range, 17)
table_prices = np.append(table_prices, [lower_choice, upper_choice])
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

# Highlighting the columns for lower_choice and upper_choice
for col in range(len(table_prices)):
    if table_prices[col] == lower_choice:
        table[(0, col)].set_facecolor('#FFB6C1')
        table[(1, col)].set_facecolor('#FFB6C1')
        table[(2, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == upper_choice:
        table[(0, col)].set_facecolor('#33CC33')
        table[(1, col)].set_facecolor('#33CC33')
        table[(2, col)].set_facecolor('#33CC33')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
