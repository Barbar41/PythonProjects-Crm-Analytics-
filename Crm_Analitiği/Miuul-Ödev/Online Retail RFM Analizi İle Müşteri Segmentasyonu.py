################
# RFM Analizi İle MüşteriSegmentayonu
################

#########################
# iş Problemi
########################

# İngiltere merkezli perakende şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istemektedir.
# Ortak davranışlar sergileyen müşteri segmentleri özelinde pazarlama çalışmaları yapmanın gelir artışı sağlayacağını düşünmektedir.
# Segmentlere ayırmak için RFM analizi kullanılacaktır.

# Veri Seti Hikayesi

# Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
# arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi mevcuttur.

# 8 Değişken 541.909 Gözlem 45.6MB
#######################################

# InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
# StockCode Ürün kodu ( Her bir ürün için eşsiz )
# Description Ürün ismi
# Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
# InvoiceDate Fatura tarihi
# UnitPrice Fatura fiyatı ( Sterlin )
# CustomerID Eşsiz müşteri numarası
# Country Ülke ismi

##############################################
# Görev 1: Veriyi Anlama ve Hazırlama
#############################################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows',None) satırlar kalabalık olacagı ıcın gerek yok
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Sayısal basamagın vırgulden sonrakı kac basamagı

# Adım 1: Online Retail II excelindeki 2010-2011 verisini okuyunuz. Oluşturduğunuz dataframe’in kopyasını oluşturunuz.
df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape

##############################################################################
# En çok sipariş edilen ürün hangisi?
df.groupby("Description").agg({"Quantity": "sum"}).head()

# Toplam kaç adet eşsiz fatura kesilmiştir?
df["Invoice"].nunique()



################################################################################

# Adım 2: Veri setinin betimsel istatistiklerini inceleyiniz.
df.describe().T

# Adım 3: Veri setinde eksik gözlem var mı? Varsa hangi değişkende kaç tane eksik gözlem vardır?
df.isnull().sum()

# Adım 4: Eksik gözlemleri veri setinden çıkartınız. Çıkarma işleminde ‘inplace=True’ parametresini kullanınız.
df.dropna(inplace=True)
df.shape
df.describe().T


# Adım 5: Eşsiz ürün sayısı kaçtır?
df["Description"].nunique()

# Adım 6: Hangi üründen kaçar tane vardır?
df["Description"].value_counts().head()

# Adım 7: En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head(5)

# Adım 8: Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df[~df["Invoice"].str.contains("C", na=False)]

# Adım 9: Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz
df["TotalPrice"] = df["Quantity"] * df["Price"]

##############################################
# Görev 2: RFM Metriklerinin Hesaplanması
##############################################
df.head()
df["InvoiceDate"].max()
today_date = dt.datetime(2011 , 12, 11)
type(today_date)

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
#--Recency:Analizin yapıldığı tarih-ilgili müşterinin son satın alma tarihi
#--Frequency:İlgili müşterinin yaptığı toplam satın alma sayısı
#--Monetary: İlgili müşterinin satın alımlar sonucu bırakması gereken parasal değer


# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini groupby, agg ve lambda ile hesaplayınız.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T

############################################################################
#----Recency değeri için bugünün tarihini (2011, 12, 11) olarak kabul ediniz.
# Rfm dataframe’ini oluşturduktan sonra veri setini "monetary>0" olacak şekilde filtreleyiniz.
rfm = rfm[rfm["monetary"] > 0.1]
rfm.shape

##############################################
# Görev 3: RFM Skorlarının Oluşturulması ve Tek bir Değişkene Çevrilmesi
#############################################

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5,
                               labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5, ])
rfm["frequency_score"] = pd.cut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))
rfm.head()
# Şampiyon müşterilerimiz kim
rfm[rfm["RFM_SCORE"] == "55"]

# Daha az değerli müşteriler için
rfm[rfm["RFM_SCORE"] == "11"]

###############################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################

# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
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

# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
# Bize yapı yakalamayı iki degeri yakalamayı saglayacak kod

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

##############
# Görev 5: Aksiyon Zamanı !
#############

# Adım 1: Önemli gördüğünü 3 segmenti seçiniz. Bu üç segmenti hem aksiyon kararları açısından hemde segmentlerin yapısı açısından(ortalama RFM değerleri) yorumlayınız.

rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "cant_loose"].head()

rfm[rfm["segment"] == "at_Risk"].head()

# Eğer bu müşterilerin ID lerine erişmek istersek.

rfm[rfm["segment"] == "new_customers1"].index

rfm[rfm["segment"] == "cant_loose"].index

# Bu işlemin sonucu dışarı cıkarmak--Yeni bir df oluşurup bu id leri buraya atıyoruz.
new_df=pd.DataFrame()
new_df["new_customer1_id"] = rfm[rfm["segment"] == "new_customers1"].index

# Buradaki floatları ceviriyoruz.
new_df["new_customer1_id"] = new_df["new_customer1_id"].astype(int)

# Adım 2: "Loyal Customers" sınıfına ait customer ID'leri seçerek excel çıktısını alınız.

# Bir excel yada csv formatında dısarıya cıkarmak gerekir.
new_df.to_csv("new_customers1.csv")