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
lower_range = 2000
upper_range = 4000
show_down = 2800
show_up = 3800

expiration_date = "10/11/2024"  # Expiration date of all options
IV = 0.60  # Implied Volatility for options
r = 0.01  # Risk-free rate

# Parameters for the strategy
K_strike = 3200  # Strike price for both the call sell and put buy
premium_received_call_sell = 167.14
premium_paid_put_buy = 121.2
num_contracts_call = 6  # Number of contracts for call sell
num_contracts_put = 4   # Number of contracts for put buy

# Parameters for the long call
K_strike_long_call = 3100  # Strike price for the long call
premium_paid_call_buy = 237.43
num_contracts_call_long = 0  # Number of contracts for long call buy

S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_sell_today = black_scholes_call(S, K_strike, T, r, IV)
put_price_buy_today = black_scholes_put(S, K_strike, T, r, IV)
call_price_buy_long_today = black_scholes_call(S, K_strike_long_call, T, r, IV)

# Calculate the payoff for each leg of the strategy at expiration
payoff_call_sell = np.minimum(K_strike - S, 0) + premium_received_call_sell
payoff_put_buy = np.maximum(K_strike - S, 0) - premium_paid_put_buy
payoff_call_buy_long = np.maximum(S - K_strike_long_call, 0) - premium_paid_call_buy

# Total payoff at expiration
payoff_strategy = (payoff_call_sell * num_contracts_call  +
                   payoff_put_buy * num_contracts_put +
                   payoff_call_buy_long * num_contracts_call_long)

# Calculate the current payoff for today
current_payoff = (-call_price_sell_today * num_contracts_call +
                  put_price_buy_today * num_contracts_put +
                  call_price_buy_long_today * num_contracts_call_long)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_strategy, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_strike, color='blue', linestyle='--', label=f"Strike Price (Call Sell / Put Buy) = {K_strike}")
ax.axvline(K_strike_long_call, color='green', linestyle='--', label=f"Strike Price (Long Call) = {K_strike_long_call}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 15)
table_prices = np.append(table_prices, [K_strike, K_strike_long_call, show_up, show_down])
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

# Highlighting the columns for the strike prices
highlight_strikes = [K_strike, K_strike_long_call]
colors = ['#FFB6C1', '#ADD8E6']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
