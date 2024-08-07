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
upper_range = 4000
expiration_date = "08/02/2024"  # Expiration date of all options
IV = 0.65  # Implied Volatility for options
r = 0.01  # Risk-free rate

# Parameters for each leg of the strategy
K_put_sell = 3200
premium_received_put_sell = 74.63
num_contracts_put_sell = 5

K_call_sell = 3600
premium_received_call_sell = 178.58
num_contracts_call_sell = 2.5



S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
put_price_sell_today = black_scholes_put(S, K_put_sell, T, r, IV)
call_price_sell_today = black_scholes_call(S, K_call_sell, T, r, IV)


# Calculate the payoff for each leg of the strategy at expiration
payoff_put_sell = np.minimum(S - K_put_sell, 0) + premium_received_put_sell
payoff_call_sell = np.minimum(K_call_sell - S, 0) + premium_received_call_sell


# Total payoff at expiration
payoff_strategy = (payoff_put_sell * num_contracts_put_sell +
                   payoff_call_sell * num_contracts_call_sell )

# Calculate the current payoff for today
current_payoff = (-put_price_sell_today * num_contracts_put_sell +
                  -call_price_sell_today * num_contracts_call_sell)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_strategy, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_put_sell, color='blue', linestyle='--', label=f"Strike Price (Put Sell) = {K_put_sell}")
ax.axvline(K_call_sell, color='red', linestyle='--', label=f"Strike Price (Call Sell) = {K_call_sell}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 18)
table_prices = np.append(table_prices, [K_put_sell, K_call_sell])
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

# Highlighting the columns for strike prices
highlight_strikes = [K_put_sell, K_call_sell]
colors = ['#FFB6C1', '#ADD8E6', '#98FB98']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()