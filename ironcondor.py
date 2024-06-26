import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

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
expiration_date = "07/05/2024"  # Expiration date of all options
K2 = 3000  # Strike price of long put
K1 = 3200  # Strike price of short put
K4 = 3500  # Strike price of short call
K3 = 3700  # Strike price of long call
IV = 0.56  # Implied Volatility for options
premium_paid_put = 25.75  # Premium paid for long put
premium_received_put = 67.14  # Premium received for short put
premium_received_call = 78.42  # Premium received for short call
premium_paid_call = 39.52  # Premium paid for long call
num_contracts = 10.5
r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the price for the options today
call_price_K3_today = black_scholes_call(S, K3, T, r, IV)
call_price_K4_today = black_scholes_call(S, K4, T, r, IV)
put_price_K1_today = black_scholes_put(S, K1, T, r, IV)
put_price_K2_today = black_scholes_put(S, K2, T, r, IV)

# Calculate the payoff for the iron condor at expiration
payoff_short_put = np.maximum(K1 - S, 0)
payoff_long_put = np.maximum(K2 - S, 0)
payoff_long_call = np.maximum(S - K3, 0)
payoff_short_call = np.maximum(S - K4, 0)

payoff_iron_condor = (
    -payoff_short_put * num_contracts + premium_received_put * num_contracts
    + payoff_long_put * num_contracts - premium_paid_put * num_contracts
    + payoff_long_call * num_contracts - premium_paid_call * num_contracts
    - payoff_short_call * num_contracts + premium_received_call * num_contracts
)

# Calculate the current payoff for today
current_payoff = (
    -put_price_K1_today * num_contracts + premium_received_put * num_contracts
    + put_price_K2_today * num_contracts - premium_paid_put * num_contracts
    + call_price_K3_today * num_contracts - premium_paid_call * num_contracts
    - call_price_K4_today * num_contracts + premium_received_call * num_contracts
)

# Calculate the new time to expiration for tomorrow
T_tomorrow = (date1 - (today + timedelta(days=1))).days / 365.0

# Calculate the price for the options tomorrow
call_price_K3_tomorrow = black_scholes_call(S, K3, T_tomorrow, r, IV)
call_price_K4_tomorrow = black_scholes_call(S, K4, T_tomorrow, r, IV)
put_price_K1_tomorrow = black_scholes_put(S, K1, T_tomorrow, r, IV)
put_price_K2_tomorrow = black_scholes_put(S, K2, T_tomorrow, r, IV)

# Calculate the projected payoff for tomorrow
projected_payoff_tomorrow = (
    -put_price_K1_tomorrow * num_contracts + premium_received_put * num_contracts
    + put_price_K2_tomorrow * num_contracts - premium_paid_put * num_contracts
    + call_price_K3_tomorrow * num_contracts - premium_paid_call * num_contracts
    - call_price_K4_tomorrow * num_contracts + premium_received_call * num_contracts
)

# Calculate the net credit received
net_credit_received = (premium_received_put - premium_paid_put) + (premium_received_call - premium_paid_call)

# Calculate the breakeven prices
breakeven_price_low = K1 - net_credit_received
breakeven_price_high = K4 + net_credit_received

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_iron_condor, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.plot(S, projected_payoff_tomorrow, label='Projected Payoff for Tomorrow', linestyle='dotted', color='orange')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K2, color='blue', linestyle='--', label=f"Long Put Strike = {K2}")
ax.axvline(K1, color='red', linestyle='--', label=f"Short Put Strike = {K1}")
ax.axvline(K4, color='red', linestyle='--', label=f"Short Call Strike = {K4}")
ax.axvline(K3, color='blue', linestyle='--', label=f"Long Call Strike = {K3}")
ax.axvline(breakeven_price_low, color='green', linestyle='--', label=f"Breakeven Low = {breakeven_price_low:.2f}")
ax.axvline(breakeven_price_high, color='green', linestyle='--', label=f"Breakeven High = {breakeven_price_high:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table, including K1, K2, K3, K4, and breakeven prices
table_prices = np.linspace(lower_range, upper_range, 15)
table_prices = np.append(table_prices, [K1, K2, K3, K4])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_iron_condor)  # Interpolating payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices
table_projected_payoffs = np.interp(table_prices, S, projected_payoff_tomorrow)  # Interpolating projected payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2), np.round(table_projected_payoffs, 2)],
                  rowLabels=['PnL at Expiration', 'Current PnL', 'Projected PnL for Tomorrow'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for K1, K2, K3, K4, and breakeven prices
highlight_strikes = [K1, K2, K3, K4]
colors = ['#FFB6C1', '#ADD8E6', '#FFFF99', '#33CC33', '#BCD7FF', '#BCD7FF']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(2, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])
        table[(3, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col])])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
