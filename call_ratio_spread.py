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
lower_range = 1500
upper_range = 3000
lower_choice = 2350
upper_choice = 2750
Date = "10/25/2024"  # Expiration date of the options
K1 = 2700  # Strike price of long call
K2 = 2900  # Strike price of short calls (higher strike)
IV1 = 57.7  # Implied Volatility for long call
IV2 = 59.9 # Implied Volatility for short call
premium_paid_long = 151.7  # Premium paid for long call
premium_received_short = 85.3  # Premium received for short call
num_long_contracts = 1  # Number of long contracts
num_short_contracts = 4  # Number of short contracts (greater than long contracts)
r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the long call and short calls today
call_price_long = black_scholes_call(S, K1, T1, r, IV1 / 100)
call_price_short = black_scholes_call(S, K2, T1, r, IV2 / 100)

# Calculate the payoff at expiration (intrinsic value for long and short calls)
payoff_K1_T1 = np.maximum(S - K1, 0) - premium_paid_long  # Long call payoff
payoff_K2_T1 = -(np.maximum(S - K2, 0) - premium_received_short)  # Short call payoff (negative as it's sold)

# Combine the payoffs for the call ratio spread strategy
total_payoff_T1 = (payoff_K1_T1 * num_long_contracts) + (payoff_K2_T1 * num_short_contracts)

# Calculate the breakeven price (considering both long and short calls)
breakeven_price = K1 + (premium_paid_long - (premium_received_short * (num_short_contracts / num_long_contracts)))

# Calculate the current payoff (based on the Black-Scholes model prices)
current_payoff_long = call_price_long - premium_paid_long
current_payoff_short = -(call_price_short - premium_received_short)
current_total_payoff = (current_payoff_long * num_long_contracts) + (current_payoff_short * num_short_contracts)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, total_payoff_T1, label=f'Payoff at Expiration ({Date})', color='black')
ax.plot(S, current_total_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Long Call Strike = {K1}")
ax.axvline(K2, color='green', linestyle='--', label=f"Short Call Strike = {K2}")
ax.axvline(breakeven_price, color='blue', linestyle='--', label=f"Breakeven Price = {breakeven_price:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table, including K1, K2, and breakeven price
table_prices = np.linspace(lower_range, upper_range, 17)
table_prices = np.append(table_prices, [lower_choice, upper_choice])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, total_payoff_T1)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_total_payoff)  # Interpolating current payoffs at these prices

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
