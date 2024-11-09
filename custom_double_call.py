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

# Black-Scholes formula for put option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

# Define the parameters for the option strategy
lower_range = 1800
upper_range = 3500
lower_val = 2650
upper_val = 3050
expiration_date = "11/15/2024"  # Expiration date of all options

IV = 0.615  # Implied Volatility for options
K_call_long = 2700  # Strike price for the long call
K_call_short = 3050  # Strike price for the short call
premium_paid_long_call = 224.8 # Premium paid for the long call
premium_received_call = 41.98 # Premium received for short call
num_contracts_call_long = 1  # Number of contracts for the long call option
num_contracts_call_short = 4 # Number of contracts for the short call option

r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_today_short = black_scholes_call(S, K_call_short, T, r, IV)

call_price_today_long = black_scholes_call(S, K_call_long, T, r, IV)

# Calculate the payoff for the short call, long put, and long call at expiration
payoff_short_call = (premium_received_call - np.maximum(S - K_call_short, 0)) * num_contracts_call_short
payoff_long_call = (np.maximum(S - K_call_long, 0) - premium_paid_long_call) * num_contracts_call_long
payoff_strategy = payoff_short_call + payoff_long_call

# Calculate the current payoff for today
current_payoff = (- (call_price_today_short * num_contracts_call_short)
                  + (call_price_today_long * num_contracts_call_long) 
                  + (premium_received_call * num_contracts_call_short) - (premium_paid_long_call * num_contracts_call_long))

# Calculate the total premium received/paid
total_premium_received_call = premium_received_call * num_contracts_call_short

total_premium_paid_long_call = premium_paid_long_call * num_contracts_call_long
total_premium_net = total_premium_received_call - total_premium_paid_long_call


# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_strategy, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_call_short, color='red', linestyle='--', label=f"Short Call Strike Price = {K_call_short}")
ax.axvline(K_call_long, color='orange', linestyle='--', label=f"Long Call Strike Price = {K_call_long}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 15)
table_prices = np.append(table_prices, [K_call_short, K_call_long, lower_val, upper_val])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_strategy)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for K_put, K_call and breakeven prices
highlight_strikes = [lower_val, upper_val]
colors = ['#FFB6C1', '#ADD8E6', '#FFD700']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
