#################
# Customer Segmentation with RFM Analysis
#################

#############################
# business Problem
#############################

# The UK-based retail company wants to divide its customers into segments and determine marketing strategies according to these segments.
# He thinks that marketing activities specific to customer segments that exhibit common behaviors will increase revenue.
# RFM analysis will be used to segment.

# Dataset Story

# The data set named Online Retail II is from a UK-based retail company between 01/12/2009 - 09/12/2011.
It includes online sales transactions between #. The company's product catalog includes gift items and it is known that most of its customers are wholesalers.

#8 Variable 541,909 Observations 45.6MB
#######################

# InvoiceNo Invoice Number (If this code starts with C, it means that the transaction has been cancelled)
# StockCode Product code (unique for each product)
# Description Product name
# Quantity Number of products (How many of the products on the invoices were sold)
# InvoiceDate Invoice date
# UnitPrice Invoice price ( Sterling )
# CustomerID Unique customer number
# Country Country name

#######################
# Task 1: Understanding and Preparing Data
#######################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows',None) is not necessary as the rows will be crowded
pd.set_option('display.float_format', lambda x: '%.3f' % x) # How many digits after the comma of the numeric digit?

# Step 1: Read the 2010-2011 data in Online Retail II excel. Create a copy of the dataframe you created.
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\online_retail_II.xlsx",
                     sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape

####################### #############################
# What is the most ordered product?
df.groupby("Description").agg({"Quantity": "sum"}).head()

# How many unique invoices were issued in total?
df["Invoice"].nunique()



####################### #################

# Step 2: Examine the descriptive statistics of the data set.
df.describe().T

# Step 3: Are there any missing observations in the data set? If so, how many missing observations are there in which variable?
df.isnull().sum()

# Step 4: Remove missing observations from the data set. Use the 'inplace=True' parameter in the subtraction process.
df.dropna(inplace=True)
df.shape
df.describe().T


# Step 5: How many unique products are there?
df["Description"].nunique()

# Step 6: How many of which product are there?
df["Description"].value_counts().head()

# Step 7: List the 5 most ordered products from most to least
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head(5)

# Step 8: 'C' on the invoices indicates canceled transactions. Remove canceled transactions from the data set.
df[~df["Invoice"].str.contains("C", na=False)]

# Step 9: Create a variable called 'TotalPrice' that represents the total earnings per invoice
df["TotalPrice"] = df["Quantity"] * df["Price"]

#######################
# Task 2: Calculating RFM Metrics
#######################
df.head()
df["InvoiceDate"].max()
today_date = dt.datetime(2011 , 12, 11)
type(today_date)

# Step 1: Define Recency, Frequency and Monetary.
#--Recency: Date of analysis - last purchase date of the relevant customer
#--Frequency: Total number of purchases made by the relevant customer
#--Monetary: The monetary value that the relevant customer must leave as a result of purchases


# Step 2: Calculate customer-specific Recency, Frequency and Monetary metrics with groupby, agg and lambda.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                      'Invoice': lambda Invoice: Invoice.nunique(),
                                      'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

# Step 3: Assign the metrics you calculated to a variable named rfm.

# Step 4: Change the names of the metrics you created as recency, frequency and monetary.
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T

####################### #############################
#----For the reliability value, accept today's date as (2011, 12, 11).
# After creating the Rfm dataframe, filter the data set to "monetary>0".
rfm = rfm[rfm["monetary"] > 0.1]
rfm.shape

#######################
# Task 3: Creating RFM Scores and Converting them into a Single Variable
#######################

# Step 1: Convert Recency, Frequency and Monetary metrics into scores between 1-5 with the help of qcut.
# Step 2: Save these scores as recency_score, frequency_score and monetary_score.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5,
                                labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5, ])
rfm["frequency_score"] = pd.cut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Step 3: Express recency_score and frequency_score as a single variable and save as RF_SCORE
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                     rfm["frequency_score"].astype(str))
rfm.head()
# Who are our champion customers
rfm[rfm["RFM_SCORE"] == "55"]

# For less valuable customers
rfm[rfm["RFM_SCORE"] == "11"]

####################
# Task 4: Defining RF Score as a Segment
####################

# Step 1: Make segment definitions for the created RF scores.
seg_map = {
     r'[1-2][1-2]': 'hibernating',
     r'[1-2][3-4]': 'at_Risk',
     r'[1-2]5': 'cant_loose',
     r'3[1-2]': 'about_to_sleep',
     r'33': 'need_attention',
     r'[3-4][4-5]': 'loyal_customers',
     r'41': 'promising',
     r'51': 'new_customers1',
     r'[4-5][2-3]': 'potential_loyalists',
     r'5[4-5]': 'champions'
}

# Step 2: Convert the scores into segments with the help of the seg_map below.
# Code that will allow us to capture the structure and capture two values

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

##############
# Mission 5: Action Time!
##############

# Step 1: Select the 3 segments you consider important. Interpret these three segments both in terms of action decisions and the structure of the segments (average RFM values).

rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "cant_loose"].head()

rfm[rfm["segment"] == "at_Risk"].head()

# If we want to access the IDs of these customers.

rfm[rfm["segment"] == "new_customers1"].index

rfm[rfm["segment"] == "cant_loose"].index

# Extracting the result of this operation - We create a new df and assign these ids here.
new_df=pd.DataFrame()
new_df["new_customer1_id"] = rfm[rfm["segment"] == "new_customers1"].index

# We are converting the floats here.
new_df["new_customer1_id"] = new_df["new_customer1_id"].astype(int)

# Step 2: Select the customer IDs belonging to the "Loyal Customers" class and get the excel output.

# It is necessary to export it in an excel or csv format.
new_df.to_csv("new_customers1.csv")
