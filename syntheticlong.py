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
show_down = 2450
show_up = 2750

expiration_date = "10/04/2024"  # Expiration date of the options
IV = 52.5  # Implied Volatility for options
r = 0.01  # Risk-free rate

# Parameters for the synthetic long stock strategy
K_strike = 2600  # Strike price for both the call and put
premium_paid_call = 120.8 # Premium paid for buying the call
premium_received_put = 65.8 # Premium received for selling the put
num_contracts = 1  # Number of contracts for both call and put

S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the options
date1 = datetime.strptime(expiration_date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the current option prices using Black-Scholes formula
call_price_today = black_scholes_call(S, K_strike, T, r, IV/100)
put_price_today = black_scholes_put(S, K_strike, T, r, IV/100)

# Calculate the payoff for each leg of the strategy at expiration
payoff_call_long = np.maximum(S - K_strike, 0) - premium_paid_call
payoff_put_short = premium_received_put - np.maximum(K_strike - S, 0)

# Total payoff at expiration
payoff_strategy = (payoff_call_long + payoff_put_short) * num_contracts

# Calculate the current profit/loss of the strategy
current_payoff = ((call_price_today - premium_paid_call) + 
                  (premium_received_put - put_price_today)) * num_contracts

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, payoff_strategy, label=f'Payoff at Expiration ({expiration_date})', color='black')
ax.plot(S, current_payoff, label='Current Profit/Loss', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_strike, color='blue', linestyle='--', label=f"Strike Price = {K_strike}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table
table_prices = np.linspace(lower_range, upper_range, 15)
table_prices = np.append(table_prices, [K_strike, show_up, show_down])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, payoff_strategy)
table_current_payoffs = np.interp(table_prices, S, current_payoff)

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Profit/Loss at Expiration', 'Current Profit/Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])

# Highlighting the columns for the strike price
highlight_strikes = [show_down, show_up]
colors = ['#FFB6C1']
for col in range(len(table_prices)):
    if table_prices[col] in highlight_strikes:
        table[(0, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col]) % len(colors)])
        table[(1, col)].set_facecolor(colors[highlight_strikes.index(table_prices[col]) % len(colors)])

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
