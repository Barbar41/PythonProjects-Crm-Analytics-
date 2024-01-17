#######################
# CLTV Estimation with BG-NBD and Gamma-Gamma
#######################

#######################
# Business Problem
#######################
# FLO wants to determine a roadmap for sales and marketing activities.
# In order for the company to make medium-long term plans, it is necessary to estimate the potential value that existing customers will provide to the company in the future.
#######################
####################
# Dataset Story
####################
# The data set consists of information obtained from the past shopping behavior of customers who made their last purchases from Flo via OmniChannel (both online and offline shopping) in 2020 - 2021.
#13 Variable 19,945 Observations 2.7MB
# master_id Unique customer number
# order_channel Which channel of the shopping platform is used (Android, ios, Desktop, Mobile)
# last_order_channel The channel where the last purchase was made
#first_order_date The date of the customer's first purchase
# last_order_date Customer's last purchase date
# last_order_date_online The last shopping date of the customer on the online platform
# last_order_date_offline The last shopping date of the customer on the offline platform
# order_num_total_ever_online Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline Total number of purchases made by the customer offline
# customer_value_total_ever_offline Total price paid by the customer for offline purchases
# customer_value_total_ever_online Total price paid by the customer for online purchases
# interested_in_categories_12 List of categories the customer has shopped in the last 12 months

####################### ##############
# Task 1: Preparing the Data
####################### #############

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

# Show All Columns
pd.set_option("display.max_columns",None)

# Show all lines
# pd.set_option("display.max_rows", None)

# Take two digits after the comma
pd.set_option("display.float_format", lambda x:"%3.f" % x)
pd.options.mode.chained_assignment =None

# Step1: Read flo_data_20K.csv data.
df_=pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analysis\datasets\flo_data_20k.csv")
df= df_.copy()
df.head()

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


# Step3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"
#--Suppress variables if they have outliers.
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
     replace_with_thresholds(df, col)

# Step4: Omnichannel means that customers shop both online and offline platforms. Create new variables for the total number of purchases and expenditures of each customer.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# Step5: Examine the variable types. Change the type of variables expressing date to date.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

####################### #
# Task 2: Creating the CLTV Data Structure
#######################

# Step1: Take 2 days after the date of the last purchase in the data set as the analysis date.
df["last_order_date"].max() #2021-05-30
analysis_date= dt.datetime(2021,6,1)

# Step2: Create a new cltv dataframe containing customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg values.
#---Monetary value will be expressed as the average value per purchase, and recency and tenure values will be expressed in weekly basis.
cltv_df= pd.DataFrame()
cltv_df["customer_id"]= df["master_id"]

#---Time since the last purchase, specific to the customer, how many weeks has the customer not made a purchase?
cltv_df["recency_cltv_weekly"]= ((df["last_order_date"]- df["first_order_date"]).astype("timedelta64[D]")) / 7

#---Customer age
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) / 7

#---Total number of purchases
cltv_df["frequency"]= df["order_num_total"]

#---Average earnings per order
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()
####################
# Task 3: Establishing BG/NBD, Gamma-Gamma Models and Calculating CLTV
#######################
#BG/NBD models the number of purchases (purchase frequency), Gamma Gamma models the average profit

# Step1: Fit the BG/NBD model.
#---We fit the model using beta and gamma distributions
#---We find the beta and gamma distribution parameters using the maximum likelihood method.
bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
         cltv_df["recency_cltv_weekly"],
         cltv_df["T_weekly"])

cltv_df.columns

#---Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
#---Customers with the most expected purchases in 3 months for all customers
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                             cltv_df["frequency"],
                                             cltv_df["recency_cltv_weekly"],
                                             cltv_df["T_weekly"])
#---Let's examine the first 10 observations
cltv_df["exp_sales_3_month"].head(10)

#---Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.
#---Customers with the most expected purchases in 3 months for all customers
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                             cltv_df["frequency"],
                                             cltv_df["recency_cltv_weekly"],
                                             cltv_df["T_weekly"])

#---Let's examine the first 10 observations
cltv_df["exp_sales_3_month"].head(10)

#---Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.
#---Let's do the same process for 6 months.
cltv_df["exp_sales_6_month"]= bgf.predict(4*6,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])


#---3. and examine the 20 people who will make the most purchases in the 6th month. Is there a difference?
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:20]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:20]

cltv_df["exp_sales_6_month"].head(10)

# Step2: Fit the Gamma-Gamma model. Estimate the average value that customers will leave and add it to the cltv dataframe as exp_average_value.
#---assigned as ggf
ggf= GammaGammaFitter(penalizer_coef=0.01)
#---Obtaining the parameters by fitting the model according to a few metrics
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

#---We assigned average _profit as a new variable
cltv_df["exp_average_value"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])
cltv_df.head()

# Step3: Calculate 6-month CLTV and add it to the dataframe with the name cltv.
#---CLTV created with customer_lifetime_value object;
cltv = ggf.customer_lifetime_value(bgf,
                                  cltv_df["frequency"],
                                  cltv_df["recency_cltv_weekly"],
                                  cltv_df["T_weekly"],
                                  cltv_df["monetary_cltv_avg"],
                                  time=6,# covers a period of 6 months
                                  freq="W",# The data entered is weekly.
                                  discount_rate=0.01)
cltv_df["cltv"]=cltv
cltv_df.head()



# Step4: Observe the 20 people with the highest Cltv values.

#---Divide all customers into 4 groups according to 6-month standardized CLTV and add the group names to the data set.
Assign it with the name #cltv_segment.

cltv_df["cltv_segment"]= pd.qcut(cltv_df["cltv"], 4, labels=["D","C","B","A"])
cltv_df.head()


####################### ####
# Task 4: Creating Segments Based on CLTV Value
####################### ####

# Step1: Divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the data set.
# Assign it with the name cltv_segment.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D","C","B","A"])
cltv_df.head()

# Step2: Make brief 6-month action suggestions to the management for 2 groups you choose among 4 groups.

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})

# Y TRY cashback for the purchase of x TRY worth of products to increase spending
# By analyzing product associations with market basket analysis, it will focus on the top N products that these customers buy the most.
# Along with these products, the frequently preferred product can be offered within the scope of NEXT BEST OFFER.
# By analyzing which categories customers prefer products from and the frequency of consumption of these products;
# (For example, let product A be a product that is consumed on average every 4 months), a sales strategy can be developed by applying a bundle system to those whose next purchase date is approaching (i.e. x% discount if you buy 2 products).


# Increasing the number of products from which the customer receives service increases customer retention. It reduces the possibility of abandonment.
# Offering cross-products with suitable offers to people interested in new products.

# Apart from the sales strategy, these customers are our valued customers and we aim to provide a personalized experience beyond being sales-oriented.
# Sending gifts to the customer's birthday celebration, considering the behavior of special people (taking into account their product habits).

# Customer anniversary celebrations such as special surprise raffles for those who come

#######################
# BONUS: Functionalize the entire process.
#######################

def create_cltv_df(dataframe):

     # Preparing the Data
     columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
                "customer_value_total_ever_online"]
     for col in columns:
         replace_with_thresholds(dataframe, col)
     df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
     df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
     dataframe =dataframe[~(dataframe["customer_value_total"]==0) | (dataframe["order_num_total"]==0)]
     date_columns = df.columns[df.columns.str.contains("date")]
     df[date_columns] = df[date_columns].apply(pd.to_datetime)

     # Creating CLTV Data Structure
     dataframe["last_order_date"].max() #2021-05-30
     analysis_date = dt.datetime(2021, 6, 1)
     cltv_df = pd.DataFrame()
     cltv_df["customer_id"] = df["master_id"]
     cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
     cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
     cltv_df["frequency"] = df["order_num_total"]
     cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
     cltv_df= cltv_df[(cltv_df["frequency"] > 1)]

     # Establishing the BG-NBD Model
     bgf = BetaGeoFitter(penalizer_coef=0.001)
     bgf.fit(cltv_df["frequency"],
             cltv_df["recency_cltv_weekly"],
             cltv_df["T_weekly"])
     cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                                cltv_df["frequency"],
                                                cltv_df["recency_cltv_weekly"],
                                                cltv_df["T_weekly"])
     cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                                cltv_df["frequency"],
                                                cltv_df["recency_cltv_weekly"],
                                                cltv_df["T_weekly"])

     # Establishing the Gamma-Gamma Model
     ggf=GammaGammaFitter(penalizer_coef=0.01)
     ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
     cltv_df["exp_average_value"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

     #CLTV Forecast
     cltv = ggf.customer_lifetime_value(bgf,
                                        cltv_df["frequency"],
                                        cltv_df["recency_cltv_weekly"],
                                        cltv_df["T_weekly"],
                                        cltv_df["monetary_cltv_avg"],
                                        time=6, # covers a period of 6 months
                                        freq="W", # The data entered is weekly.
                                        discount_rate=0.01)
     cltv_df["cltv"] = cltv

    # CLTV segmentation
     cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
     return cltv_df

cltv_df=create_cltv_df(df)

cltv_df.head(10)