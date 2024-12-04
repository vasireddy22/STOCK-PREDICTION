import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load historical stock price data
stock_data = pd.read_csv("C:\\Users\\shabana\\OneDrive\\Desktop\\programs\\stock_data.csv")
# Remove extra spaces from column names
stock_data.columns = stock_data.columns.str.strip()

# Specify features
features = ['Open', 'High', 'Low', 'Prev. Close', 'Change', '% Change']

pair_plot = sns.pairplot(stock_data[features], diag_kind='kde')

for i, feature in enumerate(features):
    pair_plot.axes[i, i].annotate(feature, (0.5, 0.5), xycoords='axes fraction',
                                  ha='center', va='center', fontsize=30, color='blue', rotation=45)

plt.show()