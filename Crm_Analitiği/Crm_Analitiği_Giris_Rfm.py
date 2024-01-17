####################### ##############
# Customer Segmentation with RFM
####################### ##############

# 1. Business Problem
# 2. Data Understanding
# 3. Data Preparation
# 4. Calculating RFM Metrics
# 5. Calculating RFM Scores
# 6. Creating & Analyzing RFM Segments
#7. Functionalization of the Entire Process

####################### ##############
# 1. Business Problem
####################### ##############

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

####################### ##############
# 2. Data Understanding
####################### ##############

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows',None) is not necessary as the rows will be crowded
pd.set_option('display.float_format', lambda x: '%.3f' % x) # How many digits after the comma of the numeric digit?

df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\online_retail_II.xlsx",
                     sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape

# dataset row column
df.isnull().sum() # missing values check

# Accessing the number of unique values
df["Description"].nunique()

# How many of which product are there?
df["Description"].value_counts().head()

# What is the most ordered product?
df.groupby("Description").agg({"Quantity": "sum"}).head()

# Let's sort from largest to smallest
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# How many unique invoices were issued in total?
df["Invoice"].nunique()

# How much total was earned from the products per invoice?
df["TotalPrice"] = df["Quantity"] * df["Price"]

# What is the total amount spent per invoice? How much was paid in total per invoice?
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

####################### ##############
# 3. Data Preparation
####################### ##############

df.shape
df.isnull().sum()

# We make it permanent by deleting the missing values with "inplace"
df.dropna(inplace=True)
df.shape
df.describe().T

# We need to remove returned invoices from the data set
df[~df["Invoice"].str.contains("C", na=False)]

# If we wanted to choose...
df = df[df["Invoice"].str.contains("C", na=False)]

####################### ##############
# 4. Calculating RFM Metrics
####################### ##############

# Recency, Frequency, Monetary
# It will be necessary to calculate the above values for each customer.

df.head()

# We must define the day on which we do the analysis.
df["InvoiceDate"].max()

# For example, recency can be done after 2 days. This will allow us to get a difference in terms of time.
today_date = dt.datetime(2010, 12, 11)
type(today_date)

# For calculation based on all customers.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                      'Invoice': lambda Invoice: Invoice.nunique(),
                                      'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

# To define dataframe column names
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

# There shouldn't be a zero in the # monetary value, let's eliminate it.
rfm = rfm[rfm["monetary"] > 0]
rfm.shape

####################### ##############
# 5. Calculating RFM Scores
####################### ##############

# For Recency score
rfm["recency_score"] = pd.qcut(rfm["recency"], 5,
                                labels=[5, 4, 3, 2, 1]) # qcut function divides into certain pieces in order from smallest to largest

# Divide the numbers between 0-100 by 5 -->> Divide the numbers as 0-20, 20-40, 40-60, 60-80, 80-100 and label them with 5 for the smaller ones and the rest in order.

# For Monetary score
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5, ])

# For Frequency score
rfm["frequency_score"] = pd.cut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# We need to create a score variable based on these values. R AND F must be together. We calculated the M value to observe it.
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                     rfm["frequency_score"].astype(str))

rfm.describe().T

# Who are our champion customers
rfm[rfm["RFM_SCORE"] == "55"]

# For less valuable customers
rfm[rfm["RFM_SCORE"] == "11"]

####################### ##############
# 6. Creating & Analyzing RFM Segments
####################### ##############

#regex

# Segment Dic. To create
seg_map = {
     r'[1-2][1-2]': 'hibernating',
     r'[1-2][3-4]': 'at_Risk',
     r'[1-2]5': 'cant_loose',
     r'3[1-2]': 'about_to_sleep',
     r'33': 'need_attention',
     r'[3-4][4-5]': 'loyal_customers',
     r'41': 'promising',
     r'51': 'new_customers',
     r'[4-5][2-3]': 'potential_loyalists',
     r'5[4-5]': 'champions'
}

# Code that will allow us to capture the structure and capture two values

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# It is necessary to analyze the created segments.

# You may want to access the information of the people in the classes. These are not metrics and scores, we will compare them by segment by taking their averages.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

We want to focus on the #need_attention,cant loose or At_Risk class.
rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "cant_loose"].head()

rfm[rfm["segment"] == "at_Risk"].head()

# If we want to access the IDs of these customers.

rfm[rfm["segment"] == "new_customers"].index

rfm[rfm["segment"] == "cant_loose"].index

# Extracting the result of this operation - We create a new df and assign these ids here.
new_df=pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# We are converting the floats here.
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# It is necessary to export it in an excel or csv format.
new_df.to_csv("new_customers.csv")

rfm.to_csv("rfm.csv")

####################### ##############
#7. Functionalization of the Entire Process
####################### ##############

df= df_.copy()
rfm_new = create_rfm(df, csv=True)

def create_rfm(dataframe, csv=False):

     # PREPARING THE DATA
     dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
     dataframe.dropna(inplace=True)
     dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

     # CALCULATION OF RFM METRICS
     today_date = dt.datetime(2011, 12, 11)
     rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                 'Invoice': lambda num: num.nunique(),
                                                 "TotalPrice": lambda price: price.sum()})
     rfm.columns = ['recency', 'frequency', "monetary"]
     rfm = rfm[(rfm['monetary'] > 0)]

     # CALCULATION OF RFM SCORES
     rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
     rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
     rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

     # cltv_df scores were converted to categorical values and added to df
     rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                         rfm['frequency_score'].astype(str))

     # NAMING THE SEGMENTS
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

