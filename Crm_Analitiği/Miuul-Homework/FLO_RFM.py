
#######################
# Task 1: Understanding and Preparing Data
#######################

import pandas as pd
import datetime as dt
# Show All Columns
pd.set_option("display.max_columns",None)

# Show all lines
# pd.set_option("display.max_rows", None)

# Take two digits after the comma
pd.set_option("display.float_format", lambda x:"%2.f" % x)

# Show invisible columns - up to 1000 characters
pd.set_option("display.width",100)

# Step 1: Read the flo_data_20K.csv data. Create a copy of the dataframe.
df_=pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\flo_data_20k.csv")
df= df_.copy()
df.head()

# Step 2: In the data set
# a. First 10 observations,
df.head(10)

#b. variable names,
df.columns
df.shape

#c. descriptive statistics,
df.describe().T

#  D. null value,
df.isnull().sum()

#e. Examine variable types.
df.info()

# Step 3: Omnichannel means that customers shop both online and offline platforms. Total for each customer
# Create new variables for the number of purchases and spending.

##---Total number of purchases of each customer = online + offline
df["order_num_total"]= df["order_num_total_ever_online"]+ df["order_num_total_ever_offline"]
df["order_num_total"].head()
##---Total spend of each customer = Offline Spend + Online Spend
df["customer_value_total"]= df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["customer_value_total"].head()

# Step 4: Examine the variable types. Change the type of variables expressing date to date.

##---Observing variable types
df.info() #--Since some date variables appear categorical, we must change the variable type.

##---Columns containing dates were transferred to data columns
date_columns=df.columns[df.columns.str.contains("date")]

##---Type assignment was made to these variables containing dates, and a dataframe was added.
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


# Step 5: Look at the distribution of the number of customers, total number of products purchased and total expenses in shopping channels.

##---In which shopping channel: 1-How many customers are there, 2-How many products were purchased in total, 3-How much was spent in total?
df.groupby("order_channel").agg({"master_id":"count",
                                  "order_num_total":"sum",
                                  "customer_value_total":"sum"})

# Step 6: List the top 10 customers who bring the most profit.

##--The variable we created as Customer value total is online+offline.
df.sort_values("customer_value_total", ascending=False)[:10]

# Step 7: List the top 10 customers who placed the most orders.
df.sort_values("order_num_total", ascending=False)[:10]

# Step 8: Functionalize the data preparation process.
def data_prep(dataframe):
     dataframe["order_num_total"]= dataframe["order_num_total_ever_online"]+dataframe["order_num_total_ever_offline"]
     dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
     date_columns= dataframe.columns[dataframe.columns.str.contains("date")]
     dataframe[date_columns]= dataframe[date_columns].apply(pd.to_datetime)
     return df

#######################
# Task 2: Calculating RFM Metrics
#######################
# Step 1: Define Recency, Frequency and Monetary.
# Step 2: Calculate Recency, Frequency and Monetary metrics for the customer.
# Step 3: Assign the metrics you calculated to a variable named rfm.
# Step 4: Change the names of the metrics you created to recency, frequency and monetary
# -----To calculate the reliability value, you can select the analysis date 2 days after the maximum date

# The analysis date is 2 days after the date of the last purchase in the data set.
# Let's say we did the analysis 2 days after the last shopping date. That's why we added 2 days to the last date.
df["last_order_date"].max() # 2021-05-30

# Creating analysis date with datetime library
analysis_date= dt.datetime(2021,6,1)

# A new rfm dataframe containing Customer_ID, Recency, Frequency and Monetary values

# Assign an empty dataframe
rfm=pd.DataFrame()

# Adding customer IDs
rfm["customer_id"]= df["master_id"]

# Creating a new variable for the Recency metric that will subtract the last order date from the analysis date
rfm["recency"] = (analysis_date- df["last_order_date"]).astype("timedelta64[D]")#astype difference between timedelta D and date time in days

# Frequency metric is the customer's total purchases
rfm["frequency"]= df["order_num_total"]

# Monetary value left by the customer
rfm["monetary"]= df["customer_value_total"]

# check dataframe containing customer_id and recency frequency monetary metrics
rfm.head()

####################
# Task 3: Calculating RF Score
####################

# Step 1: Convert Recency, Frequency and Monetary metrics into scores between 1-5 with the help of qcut.
# Step 2: Save these scores as recency_score, frequency_score and monetary_score.

# We expect recency to be small and frequency and monetary to be large.

# qcut: Sort the values of the relevant variable from smallest to largest, divide it into 5 parts according to quarters and name them according to labels
rfm["recency_score"]=pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# We use the rank method for problems that may occur in frequencies.
rfm["frequency_score"]= pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"]= pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Let's check
rfm.head()

# Step 3: Express recency_score and frequency_score as a single variable and save it as RF_SCORE.
rfm["RF_SCORE"]= (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm["RF_SCORE"].head()
#---Recency_score and frequency_score and monetary_score should be expressed as a single variable and recorded as RFM_SCORE
rfm["RFM_SCORE"]= (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))
rfm.head()

#######################
# Task 4: Defining RF Score as a Segment
#######################
# Step 1: Make segment definitions for the created RF scores.
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

# Step 2: Convert the scores into segments with the help of the seg_map below.
rfm["segment"]= rfm["RF_SCORE"].replace(seg_map,regex=True)

#############################
# Mission 5: Action Time!
#############################

# Step 1: Examine the recency, frequency and monetary averages of the segments.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# Step 2: With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below and save the customer IDs as csv.

  # a. FLO is adding a new women's shoe brand. The product prices of the included brand are available to the general customer.
  # -above your preferences. For this reason, it is desired to communicate specifically with customers who will be interested in the promotion of the brand and product sales.
  # -Customers who will be contacted specifically are loyal customers (champions, loyal_customers) and people who shop in the female category.
  # -Save the ID numbers of these customers in the csv file.
  #--Keep the IDs of RFM segment champions and loyal customers
target_segments_customer_ids= rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]

  #--we kept master id in target segments customer id and female ones in cust_ids
cust_ids=df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("WOMEN"))]["master_id"]

#--cust_ids was a pandas series, we saved it as the new brand target customer_id.csv
cust_ids.to_csv("yeni_marka_target_customer_id.csv", index=False)

cust_ids.head()

rfm.head()

  #b. Nearly 40% discount is planned for Men's and Children's products.
  # -Customers who are good customers in the past, who are interested in the categories related to this discount, but who have not shopped for a long time and should not be lost,
  # -Those who are asleep and new customers are specifically targeted.
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
target_segments_customer_ids= rfm[rfm["segment"].isin(["cant_loose","at_Risk","new_customers"])]["customer_id"]
cust_ids= df[(df["master_id"].isin(target_segments_customer_ids))& ((df["interested_in_categories_12"].str.contains("MALE"))|(df["interested_in_categories_12"].str.contains(" CHILD")))]["master_id"]

  # -Save the IDs of the customers with the appropriate profile in the csv file.
cust_ids.to_csv("discount_target_customer_ids.csv", index=False)

##############
###BONUS###
#############################
# Functionalization of the process provides benefits so that it can be used in every period.

def create_rfm(dataframe):
    # Preparing the Data
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)

    # Calculation of RFM Metrics
    df["last_order_date"].max() # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = df["master_id"]
    rfm["recency"] = (analysis_date - df["last_order_date"]).astype("timedelta64[D]")
    rfm["frequency"] = df["order_num_total"]
    rfm["monetary"] = df["customer_value_total"]

    # Calculation of RF and RFM Scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
    rfm["RFM_SCORE"] = ( rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))

    # Naming Segments
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

    rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)
    return rfm[["customer_id", "recency", "frequency", "monetary", "RF_SCORE", "RFM_SCORE", "segment"]]

    # return: Extract as a usable object

    rfm_df=create_rfm(df)






