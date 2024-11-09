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
lower_range = 1500
upper_range = 3500
lower_choice = 2650
upper_choice = 3000
Date1 = "11/15/2024"  # Expiration date of the long call
K1 = 2800  # Strike price of long call
IV1 = 56.4 # Implied Volatility for long call
premium_paid1 = 164.47 # Premium paid for long call
Date2 = "11/15/2024"  # Expiration date of the first short call
K2 = 2900  # Strike price of first short call
IV2 = 56.1 # Implied Volatility for first short call
premium_received2 = 99.69 # Premium received for first short call
Date3 = "11/15/2024"  # Expiration date of the second short call
K3 = 3000  # Strike price of second short call
IV3 = 58.3 # Implied Volatility for second short call
premium_received3 = 59.74 # Premium received for second short call
num_contracts = 2
r = 0.01  # Risk-free rate; daily yield curve rates (use expiry length); https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield
S = np.linspace(lower_range, upper_range, 400)  # Range of stock prices

# Calculate the time to expiration for the long call
date1 = datetime.strptime(Date1, "%m/%d/%Y")
today = datetime.today()
T1 = (date1 - today).days / 365.0  # Time to expiration in years for long call

# Calculate the time to expiration for the first short call
date2 = datetime.strptime(Date2, "%m/%d/%Y")
T2 = (date2 - today).days / 365.0  # Time to expiration in years for first short call

# Calculate the time to expiration for the second short call
date3 = datetime.strptime(Date3, "%m/%d/%Y")
T3 = (date3 - today).days / 365.0  # Time to expiration in years for second short call

# Calculate the price for the long call option today
call_price_today1 = black_scholes_call(S, K1, T1, r, IV1 / 100)

# Calculate the price for the first short call option today
call_price_today2 = black_scholes_call(S, K2, T2, r, IV2 / 100)

# Calculate the price for the second short call option today
call_price_today3 = black_scholes_call(S, K3, T3, r, IV3 / 100)

# Calculate the payoff for the long call at T1 expiration (intrinsic value for long call)
payoff_K1_T1 = np.maximum(S - K1, 0) - premium_paid1  # Long call (intrinsic value - premium)

# Calculate the payoff for the first short call at T2 expiration (intrinsic value for first short call)
payoff_K2_T2 = premium_received2 - np.maximum(S - K2, 0)  # First short call (premium received - intrinsic value)

# Calculate the payoff for the second short call at T3 expiration (intrinsic value for second short call)
payoff_K3_T3 = premium_received3 - np.maximum(S - K3, 0)  # Second short call (premium received - intrinsic value)

# Calculate the total payoff
total_payoff = payoff_K1_T1 * num_contracts + payoff_K2_T2 * num_contracts + payoff_K3_T3 * num_contracts

# Calculate the breakeven price
breakeven_price = (K1 + premium_paid1 - premium_received2 - premium_received3) / num_contracts

# Calculate the current payoff for today
current_payoff = call_price_today1 * num_contracts - call_price_today2 * num_contracts - call_price_today3 * num_contracts - premium_paid1 + premium_received2 + premium_received3

# Plotting the payoffs
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(S, total_payoff, label='Total Payoff at Expiration', color='black')
ax.plot(S, current_payoff, label='Current Payoff', linestyle='dotted', color='purple')
ax.set_xlabel("Stock Price")
ax.set_ylabel("Profit / Loss")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(K1, color='r', linestyle='--', label=f"Long Call Strike = {K1}")
ax.axvline(K2, color='g', linestyle='--', label=f"First Short Call Strike = {K2}")
ax.axvline(K3, color='y', linestyle='--', label=f"Second Short Call Strike = {K3}")
ax.axvline(breakeven_price, color='blue', linestyle='--', label=f"Breakeven Price = {breakeven_price:.2f}")
ax.legend(fontsize=9)
ax.grid(True)

# Selecting specific prices for the table, including K1, K2, K3, and breakeven price
table_prices = np.linspace(lower_range, upper_range, 17)
table_prices = np.append(table_prices, [lower_choice, upper_choice, K1, K2, K3, breakeven_price])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating payoffs at these prices
table_total_payoffs = np.interp(table_prices, S, total_payoff)  # Interpolating total payoffs at these prices
table_current_payoffs = np.interp(table_prices, S, current_payoff)  # Interpolating current payoffs at these prices

# Adding a table at the bottom of the plot
table = plt.table(cellText=[np.round(table_total_payoffs, 2), np.round(table_current_payoffs, 2)],
                  rowLabels=['Total Profit / Loss at Expiration', 'Current Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for lower_choice, upper_choice, K1, K2, K3, and breakeven price
for col in range(len(table_prices)):
    if table_prices[col] == lower_choice:
        table[(0, col)].set_facecolor('#FFB6C1')
        table[(1, col)].set_facecolor('#FFB6C1')
        table[(2, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == upper_choice:
        table[(0, col)].set_facecolor('#33CC33')
        table[(1, col)].set_facecolor('#33CC33')
        table[(2, col)].set_facecolor('#33CC33')
    elif table_prices[col] == K1:
        table[(0, col)].set_facecolor('#FF9999')
        table[(1, col)].set_facecolor('#FF9999')
        table[(2, col)].set_facecolor('#FF9999')
    elif table_prices[col] == K2:
        table[(0, col)].set_facecolor('#99FF99')
        table[(1, col)].set_facecolor('#99FF99')
        table[(2, col)].set_facecolor('#99FF99')
    elif table_prices[col] == K3:
        table[(0, col)].set_facecolor('#FFFF99')
        table[(1, col)].set_facecolor('#FFFF99')
        table[(2, col)].set_facecolor('#FFFF99')
    elif table_prices[col] == breakeven_price:
        table[(0, col)].set_facecolor('#99CCFF')
        table[(1, col)].set_facecolor('#99CCFF')
        table[(2, col)].set_facecolor('#99CCFF')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()