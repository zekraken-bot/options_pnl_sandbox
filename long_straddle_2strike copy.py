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
lower_val = 2338
upper_val = 2796
expiration_date = "11/1/2024"  # Expiration date of all options
strike_put = 2500  # Strike price for the long put
strike_call = 2600  # Strike price for the long call
IV = 0.485  # Implied Volatility for options
premium_paid_put = 47.68 # Premium paid for long put
premium_paid_call = 49.85  # Premium paid for long call
num_put_contracts = 12.5
num_call_contracts = 12.5

r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_today = black_scholes_call(S, strike_call, T, r, IV)
put_price_today = black_scholes_put(S, strike_put, T, r, IV)

# Calculate the payoff for the long straddle at expiration
payoff_long_put = np.maximum(strike_put - S, 0) - premium_paid_put
payoff_long_call = np.maximum(S - strike_call, 0) - premium_paid_call
payoff_long_straddle = (payoff_long_put * num_put_contracts) + (payoff_long_call * num_call_contracts)

# Calculate the current payoff for today
current_payoff = ((put_price_today - premium_paid_put) * num_put_contracts + 
                  (call_price_today - premium_paid_call) * num_call_contracts)

# Calculate the total premium paid
total_premium_paid = (premium_paid_put * num_put_contracts) + (premium_paid_call * num_call_contracts)

# Calculate the breakeven prices
breakeven_price_low = strike_put - premium_paid_put
breakeven_price_high = strike_call + premium_paid_call

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_long_straddle, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(strike_put, color='blue', linestyle='--', label=f"Put Strike Price = {strike_put}")
ax.axvline(strike_call, color='blue', linestyle='--', label=f"Call Strike Price = {strike_call}")
ax.axvline(breakeven_price_low, color='green', linestyle='--', label=f"Breakeven Low = {breakeven_price_low:.2f}")
ax.axvline(breakeven_price_high, color='green', linestyle='--', label=f"Breakeven High = {breakeven_price_high:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 18)
table_prices = np.append(table_prices, [strike_put, strike_call, lower_val, upper_val])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_long_straddle)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for strikes and breakeven prices
highlight_strikes = [strike_put, strike_call, lower_val, upper_val]
colors = ['#FFB6C1', '#ADD8E6', '#98FB98', '#FFA07A']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
