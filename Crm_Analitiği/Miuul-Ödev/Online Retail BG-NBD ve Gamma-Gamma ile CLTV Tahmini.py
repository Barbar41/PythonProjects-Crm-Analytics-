#################
# Online Retail CLTV Forecast with BG-NBD and Gamma-Gamma
##############

# Business Problem:
# The UK-based retail company wants to determine a roadmap for its sales and marketing activities.
# In order for the company to make medium-long term plans, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

##############
# Dataset Story
##############

# The data set named Online Retail II includes online sales transactions of a UK-based retail company between 01/12/2009 and 09/12/2011.
# The company's product catalog includes gift items and it is known that most of its customers are wholesalers.
#8 Variable 541,909 Observations 45.6MB

# InvoiceNo Invoice Number (If this code starts with C, it means that the transaction has been cancelled)
# StockCode Product code (unique for each product)
# Description Product name
# Quantity Number of products (How many of the products on the invoices were sold)
# InvoiceDate Invoice date
# UnitPrice Invoice price ( Sterling )
# CustomerID Unique customer number
# Country Country name

####################### #######################
# Task 1: Predicting 6-Month CLTV by Establishing BG-NBD and Gamma-Gamma Models

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment =None

# Step1: Read Online_retail_II 2010-2011 data
df_=pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df= df_.copy()
df.head()
df.describe().T
df.isnull().sum()

# Step2: Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.

#---Catching outliers
def outlier_thresholds(dataframe, variable):
     quartile1 = dataframe[variable].quantile(0.01)
     quartile3 = dataframe[variable].quantile(0.99)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit

#---Suppression of outliers according to specified lower and upper limits
#---Note: When calculating cltv, frequency values must be integer. Therefore, round the lower and upper limits with round().
We used the #---round method to make all values integer.
def replace_with_thresholds(dataframe, variable):
     low_limit, up_limit = outlier_thresholds(dataframe, variable)
     dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
     dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

df.info()

#############################
# Data Preprocessing
#############################

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]

# We called a function to suppress outliers.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# We will create the Total price variable. The total price paid for a product.
df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date=dt.datetime(2011, 12, 11)

# Let's create our data.
cltv_df = df.groupby('Customer ID').agg(
     {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                      lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
      'Invoice': lambda Invoice: Invoice.nunique(),
      'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


# We level up to read its readability.
cltv_df.columns = cltv_df.columns.droplevel(0)

# Let's name them
cltv_df.columns = ["Recency", "T", "Frequency", "Monetary"]

# To capture weekly values
cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]

cltv_df.describe().T

# Let's create the Frequency value to be greater than 1
cltv_df = cltv_df[(cltv_df["Frequency"] > 1)]

# Let's convert the Recency and Customer age (T) values to weekly units.
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

####################### #############
# Establishing the BG-NBD Model
####################### #############

# Step 1: Estimate 6-month CLTV for UK customers using data from 2010-2011.
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['Frequency'],
         cltv_df['Recency'],
         cltv_df['T'])

bgf.predict(4 * 6,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sum()

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                                cltv_df['Frequency'],
                                                cltv_df['Recency'],
                                                cltv_df['T'])
cltv_df["expected_purc_6_month"].sum()

plot_period_transactions(bgf)
plt.show()

# Step 2: Interpret and evaluate the results you obtained.

####################### #######################
# Task 2: CLTV Analysis of Different Time Periods

# Step 1: Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
# To see how many purchases can be made in a month
bgf.predict(4,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sort_values(ascending=True).head(10)
#--Let's record
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df['Frequency'],
                                                cltv_df['Recency'],
                                                cltv_df['T'])
# To see how many purchases can be made in twelve months
bgf.predict(4*12,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sort_values(ascending=True).head(10)
#--Let's record
cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                                cltv_df['Frequency'],
                                                cltv_df['Recency'],
                                                cltv_df['T'])

plot_period_transactions(bgf)
plt.show()
# Step 2: Analyze the 10 people with the highest CLTV in 1 month and the 10 people with the highest CLTV in 12 months.

# Who are the 10 customers from whom we expect the most purchases in 1 month?

bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df['Frequency'],
                                                         cltv_df['Recency'],
                                                         cltv_df['T']).sort_values(ascending=False).head(10)

# Who are the 10 customers we expect to buy from the most in 1 year?
bgf.conditional_expected_number_of_purchases_up_to_time(52, cltv_df['Frequency'],
                                                         cltv_df['Recency'],
                                                         cltv_df['T']).sort_values(ascending=False).head(10)

# Step 3: Is there any difference? If so, why do you think it might be?

# There is no need to build a model from scratch. You can proceed through the model created in the previous task. ATTENTION!

####################### #######################
# Task 3: Segmentation and Action Recommendations

ggf = GammaGammaFitter(penalizer_coef=0.01)

# We export the model object using the model object. We access the parameter values.
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

# What this model will offer us will be conditional average_profit values
ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"]).head(10)

# We have sent the total number of transactions and average values per transaction.

# If we want to observe in a decreasing way, the expected average profit is brought for each customer.
ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                         cltv_df["Monetary"]).sort_values(ascending=False).head(10)

# To observe these output results
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                                                              cltv_df["Monetary"])
# We do Observation and Ranking.
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# Calculation of CLTV with BG-NBD and GG model.

# Model Object
cltv = ggf.customer_lifetime_value(bgf,
                                    cltv_df["Frequency"],
                                    cltv_df["Recency"],
                                    cltv_df["T"],
                                    cltv_df["Monetary"],
                                    time=6, # 3 Months
                                    freq="W", # Frequency information of T.
                                    discount_rate=0.01)
cltv.head()

# Bring all the data together for final evaluation
# Let's solve the next problem. We converted the Customer id into a variable.
cltv = cltv.reset_index()

# We merge the two data sets according to the CusTomer Id. Cltv_df and cltv will be merged.
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# As ranking;
cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final

# We add a variable, create it with qcut and divide according to the clv value and divide it by 4. (sorting from smallest to largest)
cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10) #As we sorted from largest to smallest, they all appear to be A's.

# For Customer Lifetime Value, let's group by segment and describe it according to segments. In order to see all the values.
# Now we need to turn it into a suspension that will feed our sales and marketing activities. We found the averages of each customer's turnover. If we know what we are spending on the new customer, we can make a comparison.
# It can be compared by comparing the cost of finding a customer per customer according to the returns:

# Step 1: Divide all your customers into 4 groups (segments) according to 6-month CLTV for 2010-2011 UK customers and add the group names to the data set.

cltv_final.groupby("Segment").agg(
     {"count", "mean", "sum"}).head(10)

cltv_final.to_csv("cltv_prediction_Online.csv")


# Step 2: Make brief 6-month action suggestions to the management for 2 groups you choose among 4 groups.