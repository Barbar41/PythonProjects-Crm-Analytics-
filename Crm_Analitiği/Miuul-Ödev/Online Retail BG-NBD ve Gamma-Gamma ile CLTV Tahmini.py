###############
# Online Retail BG-NBD ve Gamma-Gamma ile CLTV Tahmini
##############

# İş Problemi:
# İngiltere merkezli perakende şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

##############
# Veri Seti Hikayesi
#############

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış işlemlerini içeriyor.
# Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.
# 8 Değişken 541.909 Gözlem 45.6MB

# InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode Ürün kodu ( Her bir ürün için eşsiz )
# Description Ürün ismi
# Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice Fatura fiyatı ( Sterlin )
# CustomerID Eşsiz müşteri numarası
# Country Ülke ismi

#########################################################################################
# Görev 1: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması

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

# Adım1: Online_retail_II 2010-2011 verisini okutunuz
df_=pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\online_retail_II.xlsx",sheet_name="Year 2010-2011")
df= df_.copy()
df.head()
df.describe().T
df.isnull().sum()

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

df.info()

#########################
# Veri Ön İşleme
#########################

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Price"] > 0]
df = df[df["Quantity"] > 0]

# Aykırı değerleri baskılamak için fonk çağırdık.
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Total price değişkenini yaratacagız bir ürün ıcın odenen toplam bedel
df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date=dt.datetime(2011, 12, 11)

# Verimizi yaratalım.
cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})


# Okunulabilirliğini okumak için level sılıyoruz.
cltv_df.columns = cltv_df.columns.droplevel(0)

# İsimlendirmelerini yapalım
cltv_df.columns = ["Recency", "T", "Frequency", "Monetary"]

# Haftalık değerleri yakalamak için
cltv_df["Monetary"] = cltv_df["Monetary"] / cltv_df["Frequency"]

cltv_df.describe().T

# Frequency değerini 1 den buyuk olacak sekılde olusturalım
cltv_df = cltv_df[(cltv_df["Frequency"] > 1)]

# Recency ve Müşteri yaşı(T) değerlerini haftalık cinse çevirelim.
cltv_df["Recency"] = cltv_df["Recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

##############################################################
#  BG-NBD Modelinin Kurulması
##############################################################

# Adım 1: 2010-2011 yıllarındaki veriyi kullanarak İngiltere’deki müşteriler için 6 aylık CLTV tahmini yapınız.
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

# Adım 2: Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz.

########################################################################################
# Görev 2: Farklı Zaman Periyotlarından Oluşan CLTV Analizi

# Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# Bir ay içinde ne kadar satın alma olabileceğini görebilmek için
bgf.predict(4,
            cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sort_values(ascending=True).head(10)
#--Kayıt edelim
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['Frequency'],
                                               cltv_df['Recency'],
                                               cltv_df['T'])
# On iki ay içinde ne kadar satın alma olabileceğini görebilmek için
bgf.predict(4*12,
            cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sort_values(ascending=True).head(10)
#--Kayıt edelim
cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['Frequency'],
                                               cltv_df['Recency'],
                                               cltv_df['T'])

plot_period_transactions(bgf)
plt.show()
# Adım 2: 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?

bgf.conditional_expected_number_of_purchases_up_to_time(4, cltv_df['Frequency'],
                                                        cltv_df['Recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# 1 yıl içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
bgf.conditional_expected_number_of_purchases_up_to_time(52, cltv_df['Frequency'],
                                                        cltv_df['Recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# Adım 3: Fark var mı? Varsa sizce neden olabilir?

# Sıfırdan model kurulmasına gerek yoktur. Önceki görevde oluşturulan model üzerinden ilerlenebilir.DİKKAT!

#######################################################################################
# Görev 3: Segmentasyon ve Aksiyon Önerileri

ggf = GammaGammaFitter(penalizer_coef=0.01)

# Model nesenesını kullanarak model nesnesını verıyoruz.Parametre degerlerıne erısıyoruz.
ggf.fit(cltv_df["Frequency"], cltv_df["Monetary"])

# Bu modelın bıze suancagı sey koseullu average_profit degerlerı olacak
ggf.conditional_expected_average_profit(cltv_df["Frequency"], cltv_df["Monetary"]).head(10)

# Toplam işlem sayısını ve işlem başına ortalama değerleri göndermiş olduk.

# Azalan bir şekilde gözlemlemek istersek beklenen ortalama karı her bır musterı ıcın getırdı.
ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                        cltv_df["Monetary"]).sort_values(ascending=False).head(10)

# Bu çıktı sonuçlarını gözlemleyebilmek için
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["Frequency"],
                                                                             cltv_df["Monetary"])
# Gözlemleme ve Sıralaması yapıyoruz.
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.

# Model Nesnesi
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   time=6,  # 3 Aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv.head()

# Bütün veriyi bir araya geitrip nihai değerlendirme için
# Next problemi çözümü yapalım.Customer id değişkene çevirdik.
cltv = cltv.reset_index()

# İki veri setini CusTomer Id ye göre birleştiriyoruz.Cltv_df ile cltv birleşicek.
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# Sıralaması olarak;
cltv_final.sort_values(by="clv", ascending=False).head(10)

cltv_final

# Değişken eklıyoruz qcut ile olustursun ve clv degerıne gore bolme ıslemı yapsın ve bunu 4 e bolsun.(kucukten buyuge dogru yapacktır sıralamayı)
cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(10) #uyukten kucuge gore sıraladıgımız ıcın hepsı A gozukuyor.

# Customer Lifetime Value ıcın segmentlere gore groupby alalım ve segmentlere gore betımleyelım.Bütün değerleri görebilmek adına.
# Şimdi satış ve pazarlama faalıyetlerını ebslıyecek askıyona donsuturmelıyız.Her bır musterının bırakacagı ortalamaları bulduk yenı musterı ıcın ne harcıyoruzu bılırsek su kıyaslamayı yapabılıyoruz.
# Musteri basına musterı bulma malıyetını gelen getırılere gore kıyaslayarak karsılastırılabılır:

# Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_final.groupby("Segment").agg(
    {"count", "mean", "sum"}).head(10)

cltv_final.to_csv("cltv_prediction_Online.csv")


# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

