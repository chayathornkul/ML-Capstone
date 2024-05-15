
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

# # Plot the year
year_counts = df['Year'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(year_counts.index, year_counts.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Cars per Manufacturing Year')
plt.xticks([year_counts.index[0], year_counts.index[-1]])
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# #plot the brand
company_counts = df['Make'].value_counts().sort_index()
plt.figure(figsize=(8, 8))  # Adjust the figure size as needed
plt.barh(company_counts.index, company_counts.values, color='skyblue')
plt.ylabel('Brand')
plt.xlabel('Count')
plt.title('Number of Cars per Brand')
plt.yticks(company_counts.index)
plt.gca().invert_yaxis()  # Invert the y-axis to display brands from top to bottom
plt.grid(axis='x', linestyle='--', alpha=0.6)  # Show grid lines on x-axis
plt.show()

# plot the avg price on year

avg_price_brand = df.groupby('Make')['MSRP'].mean().reset_index()

# Plot the average price for each brand
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.barh(avg_price_brand['Make'], avg_price_brand['MSRP'], color='skyblue')
plt.xlabel('Average Price')
plt.ylabel('Brand')
plt.title('Average Price of Cars per Brand')
plt.gca().invert_yaxis()  # Invert the y-axis to display brands from top to bottom
plt.grid(axis='x', linestyle='--', alpha=0.6)  # Show grid lines on x-axis
plt.yticks(fontsize=8)
plt.show()


# Plot the average price for each year
avg_price_year = df.groupby('Year')['MSRP'].mean().reset_index()

plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
plt.bar(avg_price_year['Year'], avg_price_year['MSRP'], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Average Price')
plt.title('Average Price of Cars per Year')
plt.xticks(avg_price_year['Year'], fontsize=8)  # Set font size for x-axis labels
# plt.xticks([avg_price_year.index[0], avg_price_year.index[-1]])
plt.xticks([avg_price_year['Year'].iloc[0], avg_price_year['Year'].iloc[-1]], fontsize=8)  # Show only the first and last year
plt.grid(axis='y', linestyle='--', alpha=0.6)  # Show grid lines on y-axis
plt.show()