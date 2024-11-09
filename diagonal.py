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

# Define the parameters for the option strategy
S = np.linspace(3000, 4000, 400)  # Range of stock prices
LOWER = 3456  # Lower band
UPPER = 3631  # Upper band
K1 = 3500  # Strike price of short call
K2 = 3800  # Strike price of long call
r = 0.01  # Risk-free rate
ShortDate = "11/15/2024"  # Expiration date of short call
LongDate = "11/30/2024"  # Expiration date of long call
IV1 = 53.9  # Implied Volatility for short call
IV2 = 57.6  # Implied Volatility for long call
premium_received = 145.6  # Premium received for short call
premium_paid = 94.6  # Premium paid for long call

# calc dates
date1 = datetime.strptime(ShortDate, "%m/%d/%Y")
date2 = datetime.strptime(LongDate, "%m/%d/%Y")
today = datetime.today()
# days override
#T1 = 8
#T2 = 15
T1 = (date1 - today).days  
T2 = (date2 - today).days


# Calculate the prices for the two call options with updated IVs
call_price_K1 = black_scholes_call(S, K1, (T1/365), r, (IV1/100))
call_price_K2 = black_scholes_call(S, K2, (T2/365), r, (IV2/100))

# Calculate the payoffs for a diagonal call spread at T1 expiration (intrinsic value for short call)
payoff_K1_T1 = -np.maximum(S - K1, 0) + premium_received  # Short call (intrinsic value + premium)
payoff_K2_T1 = call_price_K2 - premium_paid  # Long call (price - premium)

# Calculate the total payoff at T1
total_payoff_T1 = payoff_K1_T1 + payoff_K2_T1

# Plotting the strategy payoff at T1 with updated premiums
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, total_payoff_T1, label='Payoff at Expiration of Short Call (T1)')
ax.set_xlabel("Stock Price at T1 Expiration")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Short Call Strike = {K1}")
ax.axvline(K2, color='g', linestyle='--', label=f"Long Call Strike = {K2}")
ax.legend(fontsize=10)
ax.grid(True)

# Selecting specific prices for the table, including K1
table_prices = np.linspace(3000, 4000, 17)
table_prices = np.append(table_prices, [K1, LOWER, UPPER])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_payoffs = np.interp(table_prices, S, total_payoff_T1)  # Interpolating payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=np.round(table_payoffs, 2).reshape(1, -1),
                  rowLabels=['Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.3, 1, 0.15])  # Adjust bbox to fit within figure

# Highlighting the columns for K1, LOWER, and UPPER
for col in range(len(table_prices)):
    if table_prices[col] == K1:
        table[(0, col)].set_facecolor('#FFA07A')
    elif table_prices[col] == LOWER:
        table[(0, col)].set_facecolor('#F08080')
    elif table_prices[col] == UPPER:
        table[(0, col)].set_facecolor('#90EE90')

plt.subplots_adjust(left=0.2, bottom=0.3)
plt.show()