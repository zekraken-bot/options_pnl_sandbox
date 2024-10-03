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

# Define the parameters for the bull put spread
lower_range = 2000
upper_range = 3500
lower_choice = 2400
upper_choice = 2710
Date = "10/11/2024"  # Expiration date of the puts
K_short = 2600  # Strike price of the short put (higher strike)
K_long = 2300  # Strike price of the long put (lower strike)
IV_short = 54.6  # Implied Volatility for short put
IV_long = 66  # Implied Volatility for long put
premium_received_short = 90.1 # Premium received for short put
premium_paid_long = 15.4  # Premium paid for long put
num_contracts = 4  # Number of contracts (same for both legs)
r = 0.01  # Risk-free rate
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration
date1 = datetime.strptime(Date, "%m/%d/%Y")
today = datetime.today()
T = (date1 - today).days / 365.0  # Time to expiration in years

# Calculate the option prices today using Black-Scholes
put_price_short = black_scholes_put(S, K_short, T, r, IV_short / 100)
put_price_long = black_scholes_put(S, K_long, T, r, IV_long / 100)

# Calculate the payoff at expiration for the bull put spread
# Short put payoff (you receive premium, but may have to buy the stock at strike price if exercised)
payoff_short_put = np.where(S < K_short, -(K_short - S) + premium_received_short, premium_received_short)
# Long put payoff (you pay premium for the right to sell the stock at strike price)
payoff_long_put = np.where(S < K_long, (K_long - S) - premium_paid_long, -premium_paid_long)

# Total payoff of the bull put spread
total_payoff = (payoff_short_put + payoff_long_put) * num_contracts

# Calculate the current profit/loss based on Black-Scholes prices
current_profit_short = premium_received_short - put_price_short
current_profit_long = -(premium_paid_long - put_price_long)
current_total_profit = (current_profit_short + current_profit_long) * num_contracts

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, total_payoff, label=f'Payoff at Expiration ({Date})', color='black')
ax.plot(S, current_total_profit, label='Current Profit / Loss', linestyle='dotted', color='purple')
ax.set_xlabel("Underlying Asset Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K_short, color='red', linestyle='--', label=f"Short Put Strike = {K_short}")
ax.axvline(K_long, color='green', linestyle='--', label=f"Long Put Strike = {K_long}")
ax.legend(fontsize=9)
ax.grid(True)

# Calculate breakeven price
breakeven_price = K_short - (premium_received_short - premium_paid_long)

# Selecting specific prices for the table, including strikes and breakeven price
table_prices = np.linspace(lower_range, upper_range, 17)
table_prices = np.append(table_prices, [lower_choice, upper_choice])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, total_payoff)
table_current_profits = np.interp(table_prices, S, current_total_profit)

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_payoffs, 2), np.round(table_current_profits, 2)],
                  rowLabels=['Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=np.round(table_prices, 2),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])

# Highlighting the columns for K1 and breakeven price
for col in range(len(table_prices)):
    if table_prices[col] == lower_choice:
        table[(0, col)].set_facecolor('#FFB6C1')
        table[(1, col)].set_facecolor('#FFB6C1')
        table[(2, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == upper_choice:
        table[(0, col)].set_facecolor('#33CC33')
        table[(1, col)].set_facecolor('#33CC33')
        table[(2, col)].set_facecolor('#33CC33')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
