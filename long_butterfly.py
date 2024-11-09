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

# Define the parameters for the butterfly option strategy
lower_range = 1500
upper_range = 3500
lower_val = 2650  # Specific lower value to highlight in the table
upper_val = 3000  # Specific upper value to highlight in the table
strike_price_low = 2800  # Strike price for the lower call we buy
strike_price_middle = 3000  # Strike price for the calls we sell
strike_price_high = 3100  # Strike price for the higher call we buy
expiration_date = "11/15/2024"  # Expiration date of all options
IV = 58.5  # Implied Volatility for options
premium_paid_call_low = 167.36  # Premium paid for buying the lower call
premium_received_call = 60.58  # Premium received for selling each call
premium_paid_call_high = 39.41  # Premium paid for buying the higher call
num_contracts_sell = 2
num_contracts_buy = 1
r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_sell_today = black_scholes_call(S, strike_price_middle, T, r, IV/100)
call_price_buy_low_today = black_scholes_call(S, strike_price_low, T, r, IV/100)
call_price_buy_high_today = black_scholes_call(S, strike_price_high, T, r, IV/100)

# Calculate the payoff for the butterfly spread at expiration
payoff_sell_call = num_contracts_sell * (-np.maximum(S - strike_price_middle, 0) + premium_received_call)
payoff_buy_call_low = num_contracts_buy * (np.maximum(S - strike_price_low, 0) - premium_paid_call_low)
payoff_buy_call_high = num_contracts_buy * (np.maximum(S - strike_price_high, 0) - premium_paid_call_high)
payoff_butterfly = payoff_sell_call + payoff_buy_call_low + payoff_buy_call_high

# Calculate the current value of the options
current_payoff_sell = num_contracts_sell * (-call_price_sell_today + premium_received_call)
current_payoff_buy_low = num_contracts_buy * (call_price_buy_low_today - premium_paid_call_low)
current_payoff_buy_high = num_contracts_buy * (call_price_buy_high_today - premium_paid_call_high)
current_payoff = current_payoff_sell + current_payoff_buy_low + current_payoff_buy_high

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_butterfly, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(strike_price_middle, color='blue', linestyle='--', label=f"Middle Strike Price (Sell 2) = {strike_price_middle}")
ax.axvline(strike_price_low, color='red', linestyle='--', label=f"Lower Strike Price (Buy) = {strike_price_low}")
ax.axvline(strike_price_high, color='green', linestyle='--', label=f"Higher Strike Price (Buy) = {strike_price_high}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 19)
table_prices = np.append(table_prices, [lower_val, upper_val])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_butterfly)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for lower_val and upper_val in the table
highlight_strikes = [lower_val, upper_val]
colors = ['#FFB6C1', '#ADD8E6']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
