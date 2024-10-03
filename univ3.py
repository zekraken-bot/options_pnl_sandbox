import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd

# Constants
initial_investment = 10000
current_price = 2336
lower_bound = 2150
upper_bound = 2600
USDC_price = 1

# Initial calculations
pricing_formula = (np.sqrt(current_price) - np.sqrt(lower_bound)) / ((1/np.sqrt(current_price)) - (1/np.sqrt(upper_bound)))
amount_WETH = initial_investment / (current_price + pricing_formula * USDC_price)
amount_USDC = amount_WETH * pricing_formula

# Function to get L
def get_L():
    if current_price <= lower_bound:
        return (amount_WETH * np.sqrt(lower_bound) * np.sqrt(upper_bound)) / (np.sqrt(upper_bound) - np.sqrt(lower_bound))
    else:
        if current_price <= upper_bound: 
            return min(amount_WETH * (np.sqrt(upper_bound) * np.sqrt(current_price)) / (np.sqrt(upper_bound) - np.sqrt(current_price)), amount_USDC / (np.sqrt(current_price) - np.sqrt(lower_bound)))
        else:   
            return amount_WETH / (np.sqrt(upper_bound) - np.sqrt(lower_bound))

L = get_L()

# Functions to get new WETH and USDC amounts
def get_new_WETH_amount(new_WETH_price):
    pool_price = new_WETH_price * USDC_price
    if pool_price < lower_bound: 
        return (L / np.sqrt(lower_bound)) - (L / np.sqrt(upper_bound))
    else:
        if new_WETH_price < upper_bound: 
            return (L / np.sqrt(pool_price)) - (L / np.sqrt(upper_bound))
        else:
            return 0

def get_new_USDC_amount(new_WETH_price):
    pool_price = new_WETH_price * USDC_price
    if pool_price <= lower_bound: 
        return 0
    else:
        if pool_price < upper_bound:
            return (L * np.sqrt(pool_price)) - (L * np.sqrt(lower_bound)) 
        else:
            return (L * np.sqrt(upper_bound)) - (L * np.sqrt(lower_bound))

# Range of new WETH prices including values below the lower bound
new_WETH_prices = np.linspace(lower_bound - 200, upper_bound, 150)  # Expanded range
new_investment_worths = []

# Calculate new investment worth for each new WETH price
for price in new_WETH_prices:
    new_amount_WETH = get_new_WETH_amount(price)
    new_amount_USDC = get_new_USDC_amount(price)
    new_investment_worth = new_amount_WETH * price + new_amount_USDC
    new_investment_worths.append(new_investment_worth)

# Ensure price range for table includes lower and upper bounds
price_range = np.linspace(lower_bound - 200, upper_bound, 15)
price_range = np.append(price_range, [current_price, lower_bound])
price_range = np.unique(np.sort(price_range))  # Ensure sorted and unique values

net_worth_range = []
profit_loss_range = []

for price in price_range:
    new_amount_WETH = get_new_WETH_amount(price)
    new_amount_USDC = get_new_USDC_amount(price)
    new_investment_worth = new_amount_WETH * price + new_amount_USDC
    net_worth_range.append(new_investment_worth)
    profit_loss_range.append(new_investment_worth - initial_investment)

# Format prices, net worths, and profit/loss with commas and dollar signs
formatted_price_range = [f"${int(price):,}" for price in price_range]
formatted_net_worth_range = [f"${int(worth):,}" for worth in net_worth_range]
formatted_profit_loss_range = [f"${int(pl):,}" for pl in profit_loss_range]

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))
line, = ax.plot(new_WETH_prices, new_investment_worths, label='New Investment Worth')
ax.axvline(x=current_price, color='red', linestyle='--', label='Current Price')
ax.set_xlabel('New WETH Price')
ax.set_ylabel('New Investment Worth')
ax.legend()
ax.grid(True)

# Interactive cursor
cursor = mplcursors.cursor(line, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f'WETH Price: {sel.target[0]:.2f}\nInvestment Worth: {sel.target[1]:.2f}'))

# Adding table to the plot
table_data = [formatted_price_range, formatted_net_worth_range, formatted_profit_loss_range]
table = plt.table(cellText=table_data,
                  rowLabels=['Price', 'Net Worth', 'Profit / Loss'],
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.5, 1, 0.3])  # Adjust bbox to fit within figure

plt.subplots_adjust(left=0.2, bottom=0.5)
plt.show()
