##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################
# ZAMAN PROJEKSİYONLU OLASILIKSAL LIFETIME VALUE TAHMİNİ

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması


##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

#################Önemliler#############################
# CLTV=(Customer Value/Churn Rate) * Profit Margin
# Customer Value=Purchase Frequency*Average Order Value
# Bütün kitlenin satınalma davranışları ve bütün kitlenin işlem başına ortalama bırakacagı kazancı olasılıksal olarak modelleyip,
# --Bu olasılıksal modelin üzerine bir kişi özelliklerini girip genel kitle davranıslarından beslenerek bir tahmin işlemi yaratmak.
# Olasılıksal forma dönüşmüş Customer Value Formulasyonu
# CLTV=Expected Number of Transaction * Expected Average Profit
# İki ayrı modelleme kullanılacak
# CLTV= BG/NBD Model * Gamma Gamma Submodel

# Bir rassal değişken demek bir değişkenin belirli bir olasılık dağılımı izlediğini varsaydıgımzıda aslında olasılık dağılımı varsaydıgımzı degıskenının ortalmadısır.
# BG/NG Model--Buy till you die(satın alma ve satın almayı bırakma surecını olasılıksal olarak modellemek).
# BG/NBD Modeli,Expected Number of Transaction için iki süreci olasılıksal olarak modeller.
# Transaction Process(Buy)+ Dropout(Till You Die)


##########################
# Gerekli Kütüphane ve Fonksiyonlar
##########################

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


##Önemli##
# Kuracak odlgumuz  modeller olasılıksal ve istatistiksel modeller oldugundan dolayı;
# Bu modellerı kuraraken kullanacak oldugumuz degıskenlerın dagılımı sonucları dırek etkıleyebılecektır.
# Bundan dolayı elımızdekı degıskenlerı olusturdak sonra bu degıskenelrdekı ayrkırı degerlere dokunmak gerekır.
# Bu sebeple once aykırı degerlerı tespıt edecegız sonrasında aykırı degerlerı baskılama yontemıyle belırlemıs oldugumuz
# Aykırı degerlerı belırlı bır esık degerı ıle degsıtırecegız.

# Kendisine girilen değişken için eşik değer belirlemek olan fonk.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit( - degerler olmadıgı ıcın )
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#########################
# Verinin Okunması
#########################
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()

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

today_date = dt.datetime(2011, 12, 11)

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# Recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# Frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# Monetary: satın alma başına ortalama kazanç


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
# 2. BG-NBD Modelinin Kurulması
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['Frequency'],
        cltv_df['Recency'],
        cltv_df['T'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1, cltv_df['Frequency'],
                                                        cltv_df['Recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# Aynı sonucu predict ile alabiliriz.Fakat BG-NBD için geçerli, GAMA GAMA modeli için geçerli değildir.
bgf.predict(1,
            cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# Bir hafta içinde bütün müşteriler için beklediğimiz satın almalar için bunu  cltv_df database eklıyelım.analiz ve takip için

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['Frequency'],
                                              cltv_df['Recency'],
                                              cltv_df['T'])

# Bir ay içinde en çok satış beklediğimiz müşterilerimiz için ise:

bgf.predict(4, cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

# Bir ay için beklenen satşları kaydedelim
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['Frequency'],
                                               cltv_df['Recency'],
                                               cltv_df['T'])
# Bir ay içinde ne kadar satın alma olabileceğini görebilmek için
bgf.predict(4,
            cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sum()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################
bgf.predict(4 * 3,
            cltv_df['Frequency'],
            cltv_df['Recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['Frequency'],
                                               cltv_df['Recency'],
                                               cltv_df['T'])

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

# Bu yapılan tahminlerin başarısını nasıl değerlendirebiliriz?

plot_period_transactions(bgf)
plt.show()  # Mavi renk gercek tahmin turuncu renkler grafik

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

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

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

# Model Nesnesi
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["Frequency"],
                                   cltv_df["Recency"],
                                   cltv_df["T"],
                                   cltv_df["Monetary"],
                                   time=3,  # 3 Aylık
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

# Duzenlı olan musterı kendı ozelınde recency degerı arttıkca musterının satınalma olasılıgı yaklasıyordur.
# Cunku musterı alısverısı yaptı yaptıktan sonra kısmı churn oluyor.Drop oluyor.Bekler ve musterının satın alma ihtıyacı ortaya cıkmaya baslar.
# Bu sebeble Recency ve müşteri yası cıftlerı oldukca yuksek degerler yada bırbirıne oldukca yakın degerler.
# Yeni müşteri oldugu halde  mevcut bıraktıgı potansıyel okadar yuksek kı gerı gelecek tum verıyı ınceledıgımde goruyorum.
# Kullanıcı davranısları dıyorkı musterı yası ıle recency arasında bole bır farklılık yakınlık gozlemlenıyorsa bu durumda(diğer faktorlerın desteklemesı lazım)
# Çaprazlama sorularla erişebiliyoruz.Yaşı yuksek ve karlılıgı yuksek musterıler var.Tablo gosterıyor dırek.
# Yenı musterım potansıyel varmı monetary yuksek evet var.Eski müşteri potansıyel varmı evet var monetary dusuk frequency yuksek.


##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_final

# Değişken eklıyoruz qcut ile olustursun ve clv degerıne gore bolme ıslemı yapsın ve bunu 4 e bolsun.(kucukten buyuge dogru yapacktır sıralamayı)
cltv_final["Segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50) #uyukten kucuge gore sıraladıgımız ıcın hepsı A gozukuyor.

# Customer Lifetime Value ıcın segmentlere gore groupby alalım ve segmentlere gore betımleyelım.Bütün değerleri görebilmek adına.
# Şimdi satış ve pazarlama faalıyetlerını ebslıyecek askıyona donsuturmelıyız.Her bır musterının bırakacagı ortalamaları bulduk yenı musterı ıcın ne harcıyoruzu bılırsek su kıyaslamayı yapabılıyoruz.
# Musteri basına musterı bulma malıyetını gelen getırılere gore kıyaslayarak karsılastırılabılır:

cltv_final.groupby("Segment").agg(
    {"count", "mean", "sum"})

##############################################################
# 6. Çalışmanın Fonksiyonlaştırılması
##############################################################

def create_cltv_p(dataframe, month=3):
    #1. Veri Ön İşleme
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

    # 2. BG-NBD Modelinin Kurulması
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
    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['Frequency'], cltv_df['Monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['Frequency'],
                                                                                 cltv_df['Monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['Frequency'],
                                       cltv_df['Recency'],
                                       cltv_df['T'],
                                       cltv_df['Monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")

