#######################
# CUSTOMER LIFETIME VALUE
#######################

# 1. Data Preparation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (number of customers making multiple purchases / all customers)
# 5. Profit Margin (profit_margin = total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creating Segments
# 9. BONUS: Functionalization of All Transactions

#######################
# 1.Data Preparation
#######################

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


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\online_retail_II.xlsx",
                     sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.isnull().sum()
df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)

# We create the "Total Price" variable
df["TotalPrice"]= df["Quantity"]* df["Price"]

# Customer-specific "TotalPrice" groupby and sum will be taken according to customers
cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),# access unique invoices
                                         'Quantity': lambda x: x.sum(),
                                         'TotalPrice': lambda x: x.sum()})

# We name them
cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

#######################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
#######################

cltv_c.head()

cltv_c["total_price"] / cltv_c["total_transaction"]

# We assign it to the variable and put it into df cltv_c
cltv_c["average_order_value"]=cltv_c["total_price"] / cltv_c["total_transaction"]

#######################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
#######################

cltv_c.head()

cltv_c["total_transaction"] /cltv_c.shape[0]

# We assign it to the variable and put it into df cltv_c
cltv_c["purchase_frequency"]= cltv_c["total_transaction"] / cltv_c.shape[0]

cltv_c.shape[0]


#######################
# 4. Repeat Rate & Churn Rate (number of customers making multiple purchases / all customers)
#######################
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

#######################
# 5. Profit Margin (profit_margin = total_price * 0.10)
#######################
cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

#######################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
#######################
cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]

# Our "customer value" values, which are our main focus for each of our customers, have arrived.
# We need to correct this value.

#######################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
#######################
cltv_c["cltv"]=(cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

# To sort these values from smallest to largest or smallest to largest

cltv_c.sort_values(by="cltv", ascending=False).head()

# Let's check the highest price in this group?
cltv_c.describe().T

#######################
# 8. Creating Segments
#######################

cltv_c
# Let's sort the customers
cltv_c.sort_values(by="cltv", ascending=False).tail()

# There are many customers and how can we focus on them? Let's get a sense of the customers' values
# The room that will need to classify the customers is with the qcut function.
cltv_c["segment"]=pd.qcut(cltv_c["cltv"], 4, labels=["D","C","B","A"])

cltv_c.sort_values(by="cltv", ascending=False).head()

# We separated the customers according to their rankings, but does this make sense?
cltv_c.groupby("segment").agg({"count","mean","sum"})

# Export and present for evaluation
cltv_c.to_csv("cltc_c.csv")

#######################
# 9. BONUS: Functionalization of All Transactions
#######################

def create_cltv_c(dataframe, profit=0.10):

     # Preparing the data
     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
     dataframe = dataframe[(dataframe['Quantity'] > 0)]
     dataframe.dropna(inplace=True)
     dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
     cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                    'Quantity': lambda x: x.sum(),
                                                    'TotalPrice': lambda x: x.sum()})
     cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
     # avg_order_value
     cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
     # purchase_frequency
     cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
     # repeat rate & churn rate
     repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
     churn_rate = 1 - repeat_rate
     # profit_margin
     cltv_c['profit_margin'] = cltv_c['total_price'] * profit
     # CustomerValue
     cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])
     # Customer Lifetime Value
     cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']
     # Segment
     cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

     return cltv_c


df = df_.copy()

clv = create_cltv_c(df


