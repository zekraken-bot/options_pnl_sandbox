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
lower_range = 2000
upper_range = 3500
lower_choice = 2650
upper_choice = 3000
Date = "11/15/2024"  # Expiration date of the puts
K1 = 2700  # Strike price of long put
K2 = 2600  # Strike price of short puts (lower strike)
IV1 = 60.7  # Implied Volatility for long put
IV2 = 65.3  # Implied Volatility for short put
premium_paid_long = 27.07 # Premium paid for long put
premium_received_short = 14.4  # Premium received for short puts
num_long_contracts = 1  # Number of long contracts
num_short_contracts = 4  # Number of short contracts
r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the long put and short puts today
put_price_long = black_scholes_put(S, K1, T1, r, IV1 / 100)
put_price_short = black_scholes_put(S, K2, T1, r, IV2 / 100)

# Calculate the payoff at expiration (intrinsic value for long and short puts)
payoff_K1_T1 = np.maximum(K1 - S, 0) - premium_paid_long  # Long put payoff
payoff_K2_T1 = -(np.maximum(K2 - S, 0) - premium_received_short)  # Short put payoff (negative as it's sold)

# Combine the payoffs for the put ratio spread strategy
total_payoff_T1 = (payoff_K1_T1 * num_long_contracts) + (payoff_K2_T1 * num_short_contracts)

# Calculate the breakeven price (considering both long and short puts)
breakeven_price = (premium_paid_long - premium_received_short * (num_short_contracts / num_long_contracts)) + K1

# Calculate the current payoff (based on the Black-Scholes model prices)
current_payoff_long = put_price_long - premium_paid_long
current_payoff_short = -(put_price_short - premium_received_short)
current_total_payoff = (current_payoff_long * num_long_contracts) + (current_payoff_short * num_short_contracts)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, total_payoff_T1, label=f'Payoff at Expiration ({Date})', color='black')
ax.plot(S, current_total_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Long Put Strike = {K1}")
ax.axvline(K2, color='green', linestyle='--', label=f"Short Put Strike = {K2}")
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

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
