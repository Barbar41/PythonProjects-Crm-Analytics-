#####################################
# BG-NBD ve Gamma-Gamma ile CLTV Tahmini
#####################################

##############################################
# İş Problemi
#############################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.
####################################
############################################
# Veri Seti Hikayesi
############################################
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
# 13 Değişken 19.945 Gözlem 2.7MB
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

################################################################
# Görev 1: Veriyi Hazırlama
##############################################################

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

# Bütün Kolonları Göster
pd.set_option("display.max_columns",None)

# Bütün satırları göster
# pd.set_option("display.max_rows", None)

# Virgülden sonra iki basamak al
pd.set_option("display.float_format", lambda x:"%3.f" % x)
pd.options.mode.chained_assignment =None

# Adım1: flo_data_20K.csv verisini okuyunuz.
df_=pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\flo_data_20k.csv")
df= df_.copy()
df.head()

# Adım2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.

#---Aykırı değerlerin yakalanması
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#---Belirlenen alt ve üst sınırlara göre aykırı değerlerin baskılanması
#---Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
#---round metodunu tüm değerlerin integer olması için kullandık
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# Adım3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"
#--Değişkenlerinin aykırı değerleri varsa baskılayanız.
columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

# Adım4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# Adım5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

###################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
###############################################

# Adım1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
df["last_order_date"].max() #2021-05-30
analysis_date= dt.datetime(2021,6,1)

# Adım2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
#---Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
cltv_df= pd.DataFrame()
cltv_df["customer_id"]= df["master_id"]

#---Son satın alma üzerinden geçen zaman müşteri özelinde  müşteri kaç haftadır alışveriş yapmıyor.
cltv_df["recency_cltv_weekly"]= ((df["last_order_date"]- df["first_order_date"]).astype("timedelta64[D]")) / 7

#---Müşteri yaşı
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) / 7

#---Toplam satın alma sayısı
cltv_df["frequency"]= df["order_num_total"]

#---Siparis basina ortalama kazanç
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()
##########################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
###########################################
#BG/NBD Satın alma sayısını modeller(purchase frequency), Gamma Gamma ise average profiti

# Adım1: BG/NBD modelini fit ediniz.
#---Beta ve gama dağılımları kullanarak modeli fit ediyoruz
#---Beta,gama dağılım parametrelerini en çok olabilirlik yöntemiyle buluyoruz.
bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

cltv_df.columns

#---3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#---Bütün müşteriler için 3 aylık en cok satın alma beklenen müşteriler
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                            cltv_df["frequency"],
                                            cltv_df["recency_cltv_weekly"],
                                            cltv_df["T_weekly"])
#---İlk 10 gözlemi inceleyelim
cltv_df["exp_sales_3_month"].head(10)

#---3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#---Bütün müşteriler için 3 aylık en cok satın alma beklenen müşteriler
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                            cltv_df["frequency"],
                                            cltv_df["recency_cltv_weekly"],
                                            cltv_df["T_weekly"])

#---İlk 10 gözlemi inceleyelim
cltv_df["exp_sales_3_month"].head(10)

#---6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
#---Aynı işlem 6 ay için yapalım.
cltv_df["exp_sales_6_month"]= bgf.predict(4*6,
                                          cltv_df["frequency"],
                                          cltv_df["recency_cltv_weekly"],
                                          cltv_df["T_weekly"])


#---3. ve 6. ayda ki en çok satın alma gerçekleştiricek 20 kişiyi inceleyiniz.Fark var mı?
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:20]

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:20]

cltv_df["exp_sales_6_month"].head(10)

# Adım2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
#---ggf olarak atanması
ggf= GammaGammaFitter(penalizer_coef=0.01)
#---Modelin bir iki metriğe göre fit edilerek parametrelerin elde edilmesi
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

#---Beklenen ortalama karı(average _profit) yeni bir değişken olarak atadık
cltv_df["exp_average_value"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                      cltv_df["monetary_cltv_avg"])
cltv_df.head()

# Adım3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
#---CLTV customer_lifetime_value nesnesiyle oluşturması;
cltv = ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_avg"],
                                 time=6,# 6 aylık surecı kapsar
                                 freq="W",# Girilen veriler haftalıkdır.
                                 discount_rate=0.01)
cltv_df["cltv"]=cltv
cltv_df.head()



# Adım4: Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.

#---6 aylık standartlaştırılmış CLTV ye göre tüm müşterileri 4 gruba ayırınız ve grup isimlerini veeri setine ekleyiniz.
#cltv_segment isimi ile ataynız.

cltv_df["cltv_segment"]= pd.qcut(cltv_df["cltv"], 4, labels=["D","C","B","A"])
cltv_df.head()


######################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
######################################################

# Adım1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D","C","B","A"])
cltv_df.head()

# Adım2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})

# Harcama arttırmaya yönelik X tl lik urun alımına y Tl cashback
# Market basket analizi ile ürün birlikteliklerini analiz ederek,bu musterilerin en cok aldıkalrı top N tane ürüne odaklanacak.
# Bu ürünlerle birlikte sıklıkla tercih edilen ürün NEXT BEST OFFER kapsamına sunulabilir.
# Müşterilerin hangi aktegorilerden ürün tercihi yaptıklarına ve bu ürünlerin tüketim sıklıgını analiz ederek;
# (Örneğin A ürün ortalama 4 ayda bir tüketilen bir ürün olsun),bir sonraki satın alım tarihi yaklaşanlara bundle sistemi uygulayarak(yani 2 ürün alana%x indirim gibi)satış stratejisi geliştirilebilir.


# Müşterinin hizmet aldığı yararlandıgı urun sayısını arttırmak müşteri tutundurmayı arttırır.Terk olasılığını azaltır.
# Yeni çıkıcak ürünlerle ilgili kişilere uygun tekliflerle çapraz ürün sunulması.

# Satış strateji dışında bu müşteriler değerli müşterilerimiz ve satış odaklı olmanın dışında kişiye özel deneyim yaşatmak.
# Müşteri doğum günü kutlamasına özel kişilerin davranışları(ürün alışkanlıkları göz önüne alınarak)hediye gönderilmesi.

# Müşteri yıl dönümü kutlaması gelenlere özel süpriz çekilişler gibi

######################################
# BONUS:Tüm süreci fonksiyonlaştırınız.
######################################

def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
               "customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)
    df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    dataframe =dataframe[~(dataframe["customer_value_total"]==0) | (dataframe["order_num_total"]==0)]
    date_columns = df.columns[df.columns.str.contains("date")]
    df[date_columns] = df[date_columns].apply(pd.to_datetime)

    # CLTV Veri Yapısnın Oluşturulması
    dataframe["last_order_date"].max() #2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = df["master_id"]
    cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
    cltv_df["frequency"] = df["order_num_total"]
    cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]
    cltv_df= cltv_df[(cltv_df["frequency"] > 1)]

    # BG-NBD Modelinin Kurulması
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

    # Gamma-Gamma Modelinin Kurulması
    ggf=GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

    # CLTV Tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"],
                                       cltv_df["monetary_cltv_avg"],
                                       time=6,  # 6 aylık surecı kapsar
                                       freq="W",  # Girilen veriler haftalıkdır.
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

   # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
    return cltv_df

cltv_df=create_cltv_df(df)

cltv_df.head(10)




