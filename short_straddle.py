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
lower_range = 1700
upper_range = 3200
lower_val = 2225
upper_val = 2525
expiration_date = "11/08/2024"  # Expiration date of the options
K = 2425  # Strike price for both the short call and short put
IV = 1.03  # Implied Volatility for options
premium_received_put =  75.76 # Premium received for selling the put
premium_received_call = 85.78  # Premium received for selling the call
num_put_contracts = 0.5
num_call_contracts = 0.5

r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the current option prices using Black-Scholes formula
call_price_today = black_scholes_call(S, K, T, r, IV)
put_price_today = black_scholes_put(S, K, T, r, IV)

# Calculate the payoff for the short straddle at expiration
payoff_short_put = premium_received_put - np.maximum(K - S, 0)
payoff_short_call = premium_received_call - np.maximum(S - K, 0)
payoff_short_straddle = (payoff_short_put * num_put_contracts) + (payoff_short_call * num_call_contracts)

# Calculate the current profit/loss of the strategy
current_payoff = ((premium_received_put - put_price_today) * num_put_contracts +
                  (premium_received_call - call_price_today) * num_call_contracts)

# Calculate the total premium received
total_premium_received = (premium_received_put * num_put_contracts) + (premium_received_call * num_call_contracts)

# Calculate the breakeven prices
breakeven_price_low = K - total_premium_received / (num_put_contracts + num_call_contracts)
breakeven_price_high = K + total_premium_received / (num_put_contracts + num_call_contracts)

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_short_straddle, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Profit/Loss', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K, color='blue', linestyle='--', label=f"Strike Price = {K}")
ax.axvline(breakeven_price_low, color='green', linestyle='--', label=f"Breakeven Low = {breakeven_price_low:.2f}")
ax.axvline(breakeven_price_high, color='green', linestyle='--', label=f"Breakeven High = {breakeven_price_high:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 18)
table_prices = np.append(table_prices, [K, lower_val, upper_val, breakeven_price_low, breakeven_price_high])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_short_straddle)
table_current_payoffs = np.interp(table_prices, S, current_payoff)

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=np.round(table_prices, 2),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])

# Highlighting the columns for K and breakeven prices
highlight_strikes = [K, lower_val, upper_val]
colors = ['#FFB6C1', '#ADD8E6', '#98FB98']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        idx = highlight_strikes.index(table_prices[col]) % len(colors)
        table[(0, col)].set_facecolor(colors[idx])
        table[(1, col)].set_facecolor(colors[idx])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
