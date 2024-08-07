import numpy as np
import matplotlib.pyplot as plt

# Function to calculate profit/loss for short ETH position
def calculate_profit_loss(current_price, amount_shorted, price_range):
    prices = np.linspace(price_range[0], price_range[1], 400)
    profit_loss = (current_price - prices) * amount_shorted
    return prices, profit_loss

# Inputs
current_eth_price = 2524  # Current ETH price in dollars
amount_eth_shorted = 1  # Amount of ETH shorted

# Define the range where you want to see the profit/loss outcome
price_range = (1000, 5000)

# Prices to highlight
highlight_price_1 = 2150
highlight_price_2 = 2800

# Calculate profit/loss
prices, profit_loss = calculate_profit_loss(current_eth_price, amount_eth_shorted, price_range)

# Plotting the profit/loss
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(prices, profit_loss, label='Profit/Loss for Short ETH Position', color='black')
ax.set_xlabel("ETH Price in USD")
ax.set_ylabel("Profit / Loss in USD")
ax.axhline(0, color='black', lw=0.5)
ax.axvline(current_eth_price, color='r', linestyle='--', label=f"Current ETH Price = ${current_eth_price}")
ax.legend(fontsize=9)
ax.grid(True)

# Adding a table at the bottom of the plot
table_prices = np.linspace(price_range[0], price_range[1], 17)
table_prices = np.append(table_prices, [highlight_price_1, highlight_price_2])
table_prices = np.unique(np.sort(table_prices))  # Ensure sorted and unique values

# Interpolating profit/loss at these prices
table_profit_loss = np.interp(table_prices, prices, profit_loss)

# Creating a table for profit/loss at specific prices
table = plt.table(cellText=[np.round(table_profit_loss, 2)],
                  rowLabels=['Profit / Loss'],
                  colLabels=table_prices.astype(int),
                  cellLoc='center',
                  rowLoc='center',
                  loc='bottom',
                  bbox=[0.0, -0.4, 1, 0.25])  # Adjust bbox to fit within figure

# Highlighting the columns for the highlight prices
for col in range(len(table_prices)):
    if table_prices[col] == highlight_price_1:
        table[(0, col)].set_facecolor('#FFB6C1')
    elif table_prices[col] == highlight_price_2:
        table[(0, col)].set_facecolor('#33CC33')

plt.subplots_adjust(left=0.2, bottom=0.4)
plt.show()
