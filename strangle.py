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
lower_range = 2800
upper_range = 4200
expiration_date = "08/09/2024"  # Expiration date of all options
upper_price = 3600
lower_price = 3300
K_put = 3100  # Strike price for the short put
K_call = 3700  # Strike price for the short call
IV = 0.72  # Implied Volatility for options
premium_received_put = 33.4  # Premium received for short put
premium_received_call = 82.7  # Premium received for short call
num_contracts_put = 15  # Number of contracts for the put option
num_contracts_call = 15  # Number of contracts for the call option

r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_today = black_scholes_call(S, K_call, T, r, IV)
put_price_today = black_scholes_put(S, K_put, T, r, IV)

# Calculate the payoff for the short call and short put at expiration
payoff_short_call = (premium_received_call - np.maximum(S - K_call, 0)) * num_contracts_call
payoff_short_put = (premium_received_put - np.maximum(K_put - S, 0)) * num_contracts_put
payoff_strategy = payoff_short_call + payoff_short_put

# Calculate the current payoff for today
current_payoff = ((premium_received_put - put_price_today) * num_contracts_put +
                  (premium_received_call - call_price_today) * num_contracts_call)

# Calculate the total premium received
total_premium_received_put = premium_received_put * num_contracts_put
total_premium_received_call = premium_received_call * num_contracts_call
total_premium_net = total_premium_received_put + total_premium_received_call

# Calculate the breakeven prices
breakeven_price_low = K_put - (total_premium_received_put / num_contracts_put)
breakeven_price_high = K_call + (total_premium_received_call / num_contracts_call)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_strategy, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_put, color='blue', linestyle='--', label=f"Put Strike Price = {K_put}")
ax.axvline(K_call, color='red', linestyle='--', label=f"Call Strike Price = {K_call}")
ax.axvline(breakeven_price_low, color='green', linestyle='--', label=f"Breakeven Low = {breakeven_price_low:.2f}")
ax.axvline(breakeven_price_high, color='green', linestyle='--', label=f"Breakeven High = {breakeven_price_high:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 14)
table_prices = np.append(table_prices, [K_put, K_call, upper_price, lower_price])
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
highlight_strikes = [K_put, K_call]
colors = ['#FFB6C1', '#ADD8E6', '#98FB98']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
