####################### #############
# CLTV Prediction with BG-NBD and Gamma-Gamma
####################### #############
# PROJECTIVE PROBABORATIVE LIFETIME VALUE ESTIMATION

# 1. Data Preparation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
# 5. Creating Segments Based on CLTV
# 6. Functionalization of the work


####################### #############
# 1. Data Preparation
####################### #############

# An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.

# Dataset Story

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# The data set named Online Retail II is from a UK-based online sales store.
# Includes sales between 01/12/2009 - 09/12/2011.

# Variables

# InvoiceNo: Invoice number. Unique number for each transaction, i.e. invoice. If it starts with C, the transaction is cancelled.
# StockCode: Product code. Unique number for each product.
# Description: Product name
# Quantity: Number of products. It indicates how many of the products on the invoices were sold.
# InvoiceDate: Invoice date and time.
# UnitPrice: Product price (in Pounds Sterling)
# CustomerID: Unique customer number
# Country: Country name. The country where the customer lives.

##################Important############
# CLTV=(Customer Value/Churn Rate) * Profit Margin
# Customer Value=Purchase Frequency*Average Order Value
# By probabilistically modeling the purchasing behavior of the entire audience and the average profit per transaction of the entire audience,
# --Creating a prediction process by entering a person's characteristics into this probabilistic model and feeding on general mass behavior.
# Customer Value Formulation transformed into probabilistic form
# CLTV=Expected Number of Transaction * Expected Average Profit
# Two separate modeling will be used
# CLTV= BG/NBD Model * Gamma Gamma Submodel

# A random variable means that when we assume that a variable follows a certain probability distribution, it is actually the mean of the variable that we assume the probability distribution.
# BG/NG Model--Buy till you die (modeling the buying and stopping process probabilistically).
# BG/NBD Model probabilistically models two processes for Expected Number of Transactions.
# Transaction Process(Buy)+ Dropout(Till You Die)


#############################
# Required Libraries and Functions
#############################

# !pip install lifetimes
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


##Important##
# Since the models we will build are probabilistic and statistical models;
# The distribution of the variables we will use when building these models may directly affect the results.
# Therefore, after creating the variables we have, it is necessary to touch the outlier values in these variables.
# For this reason, we will first detect the outliers, and then we will use the method of suppressing the outliers.
# We will replace outliers with a certain threshold value.

# Function to determine the threshold value for the variable entered.
def outlier_thresholds(dataframe, variable):
     quartile1 = dataframe[variable].quantile(0.01)
     quartile3 = dataframe[variable].quantile(0.99)
     interquantile_range = quartile3 - quartile1
     up_limit = quartile3 + 1.5 * interquantile_range
     low_limit = quartile1 - 1.5 * interquantile_range
     return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
     low_limit, up_limit = outlier_thresholds(dataframe, variable)
     # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit( - since there are no values)
     dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#############################
# Reading Data
#############################
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\online_retail_II.xlsx",
                     sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

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

today_date = dt.datetime(2011, 12, 11)

#############################
# Preparation of Lifetime Data Structure
#############################

# Recency: Time since last purchase. Weekly. (user specific)
# T: Age of the customer. Weekly. (how long before the date of analysis was the first purchase made)
# Frequency: total number of recurring purchases (frequency>1)
# Monetary: average earnings per purchase


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
#2. Establishing the BG-NBD Model
####################### #############

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['Frequency'],
         cltv_df['Recency'],
         cltv_df['T'])

####################### ##############
# Who are the 10 customers from whom we expect the most purchases in a week?
####################### ##############

bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df['Frequency'],
                                                         cltv_df['Recency'],
                                                         cltv_df['T']).sort_values(ascending=False).head(10)

# We can get the same result with predict. But it is valid for BG-NBD, but not for the GAMA GAMA model.
bgf.predict(1,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sort_values(ascending=False).head(10)

# Let's add this to the cltv_df database for the purchases we expect for all customers within a week. For later analysis and tracking

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['Frequency'],
                                               cltv_df['Recency'],
                                               cltv_df['T'])

# For our customers from whom we expect the most sales in a month:

bgf.predict(4, cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sort_values(ascending=False).head(10)

# Let's record the expected sales for a month
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                cltv_df['Frequency'],
                                                cltv_df['Recency'],
                                                cltv_df['T'])
# To see how many purchases can be made in a month
bgf.predict(4,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sum()

####################### ##############
# What is the Expected Number of Sales of the Entire Company in 3 Months?
####################### ##############
bgf.predict(4 * 3,
             cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                                cltv_df['Frequency'],
                                                cltv_df['Recency'],
                                                cltv_df['T'])

####################### ##############
# Evaluation of Forecast Results
####################### ##############

# How can we evaluate the success of these predictions?

plot_period_transactions(bgf)
plt.show() # Blue color actual forecast orange colors chart

####################### #############
# 3. Establishing the GAMMA-GAMMA Model
####################### #############

ggf = GammaGammaFitter(penalizer_coef=0.01)

# We export the model object using the model object. We access the parameter values.
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

# What this model will offer us will be conditional average_profit values

ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"]).head(10)

# We have sent the total number of transactions and average values per transaction.

# If we want to observe it in a decreasing way, let's bring the expected average profit for each customer.
ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                         cltv_df["Monetary"]).sort_values(ascending=False).head(10)

# To observe these output results
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                                                              cltv_df["Monetary"])
# We do Observation and Ranking.
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

####################### #############
# 4. Calculation of CLTV with BG-NBD and GG model.
####################### #############

# Model Object
cltv = ggf.customer_lifetime_value(bgf,
                                    cltv_df["Frequency"],
                                    cltv_df["Recency"],
                                    cltv_df["T"],
                                    cltv_df["Monetary"],
                                    time=3, # 3 Months
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

# As the recency value of a regular customer increases, the customer's probability of purchasing increases.
# Because after the customer makes the purchase, the part churns. It drops. It waits and the customer's purchasing need begins to emerge.
# For this reason, Recency and customer age pairs are very high or very close to each other.
# When I examine all the data, I see that even though he is a new customer, the potential he has left is so high that he can come back.
# If user behavior indicates that there is a significant difference between customer age and reliability, then this should be supported by other factors.
# We can reach you with cross-cutting questions. There are customers who are older and have higher profitability. The table shows it directly.
# Is there new customer potential, monetary is high, yes there is. Is there old customer potential, yes there is, monetary is low frequency is high.


####################### #############
#5. Creating Segments Based on CLTV
####################### #############

cltv_final

# We add a variable, create it with qcut and divide according to the clv value and divide it by 4. (sorting from smallest to largest)

cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)
  #As we sorted them from largest to smallest, they all appear to be A's.

# For Customer Lifetime Value, let's group by segment and describe it according to segments. In order to see all the values.
# Now we need to turn sales and marketing activities into a suspension that will feed them. We found the averages of each customer's turnover. If we know what we are spending on the new customer, we can make a comparison.

# It can be compared by comparing the cost of finding a customer per customer according to the returns:

cltv_final.groupby("Segment").agg(
     {"count", "mean", "sum"})

####################### #############
#6. Functionalization of Work
####################### #############

def create_cltv_p(dataframe, month=3):
     #one. Data Preprocessing
     dataframe.dropna(inplace=True)
     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
     dataframe = dataframe[dataframe["Quantity"]>0]
     dataframe = dataframe[dataframe["Price"]> 0]
     replace_with_thresholds(dataframe, "Quantity")
     replace_with_thresholds(dataframe, "Price")
     dataframe["TotalPrice"]= dataframe["Quantity"] * dataframe["Price"]
     today_date=dt.datetime(2011, 12, 11)

     cltv_df= dataframe.groupby("Customer ID").agg({"InvoiceDate":[lambda InvoiceDate: (InvoiceDate.max()-InvoiceDate.min()).days,
                                                            lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                             "Invoice": lambda Invoice: Invoice.nunique(),
                                             "TotalPrice": lambda TotalPrice: TotalPrice.sum()})
     cltv_df.columns=cltv_df.columns.droplevel(0)
     cltv_df.columns=["Recency", "T", "Frequency", "Monetary"]
     cltv_df["Monetary"]= cltv_df["Monetary"] / cltv_df["Frequency"]
     cltv_df=cltv_df[(cltv_df["Frequency"]> 1)]
     cltv_df["Recency"]= cltv_df["Recency"] /7
     cltv_df["T"] = cltv_df["T"] /7

     #2. Establishing the BG-NBD Model
     bgf = BetaGeoFitter(penalizer_coef=0.001)
     bgf.fit(cltv_df['Frequency'],
             cltv_df['Recency'],
             cltv_df['T'])

     cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                   cltv_df['Frequency'],
                                                   cltv_df['Recency'],
                                                   cltv_df['T'])

     cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                    cltv_df['Frequency'],
                                                    cltv_df['Recency'],
                                                    cltv_df['T'])

     cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                    cltv_df['Frequency'],
                                                    cltv_df['Recency'],
                                                    cltv_df['T'])
     # 3. Establishing the GAMMA-GAMMA Model
     ggf = GammaGammaFitter(penalizer_coef=0.01)
     ggf.fit(cltv_df['Frequency'], cltv_df['Monetary'])
     cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['Frequency'],
                                                                                  cltv_df['Monetary'])

     # 4. Calculation of CLTV with BG-NBD and GG model.
     cltv = ggf.customer_lifetime_value(bgf,
                                        cltv_df['Frequency'],
                                        cltv_df['Recency'],
                                        cltv_df['T'],
                                        cltv_df['Monetary'],
                                        time=month, # 3 months
                                        freq="W", # Frequency information of T.
                                        discount_rate=0.01)

     cltv = cltv.reset_index()
     cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
     cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

     return cltv_final

df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")
