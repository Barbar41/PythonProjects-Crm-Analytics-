###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows',None) satırlar kalabalık olacagı ıcın gerek yok
pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Sayısal basamagın vırgulden sonrakı kac basamagı

df_ = pd.read_excel(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape

# dataset satır sutun
df.isnull().sum()  # eksik degerler kontrolu

# Eşsiz değer sayısına erişmek
df["Description"].nunique()

# Hangi üründen kaçar tane var?
df["Description"].value_counts().head()

# En çok sipariş edilen ürün hangisi?
df.groupby("Description").agg({"Quantity": "sum"}).head()

# Büyükten küçüğe doğru sıralayalım
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# Toplam kaç adet eşsiz fatura kesilmiştir?
df["Invoice"].nunique()

# Fatura başına ürünlerden toplam ne kadar kazanılmıştır?
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Fatura başına toplam harcanan tutar nedir?INvoice başına toplam ne kadar ödendi?
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df.shape
df.isnull().sum()

# Eksik değerleri silerek"inplace" ile kalıcılık
df.dropna(inplace=True)
df.shape
df.describe().T

# İade edilen faturaları veri setinden cıkarmamız gerek
df[~df["Invoice"].str.contains("C", na=False)]

# Seçmek isteseydik eğer..
df = df[df["Invoice"].str.contains("C", na=False)]

###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency,Frequency, Monetary
# Her bir müşteri özelinde yukarıdaki değerleri hesaplamak gerekecek.

df.head()

# Analizi yaptıgımız günü tanımlamalıyız.
df["InvoiceDate"].max()

# Örneğin 2 gun sonrası uzerınden recency işlemi yapılabilir.Zaman acısından fark alabilmemiz sağlayacak
today_date = dt.datetime(2010, 12, 11)
type(today_date)

# Bütün müşterilere göre hesaplama işlemi için.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

# Dataframe sutun ısımlerını tanımlamak
rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

# monetary degerınde sıfır olmamalı bunu ucuruyoruz
rfm = rfm[rfm["monetary"] > 0]
rfm.shape

###############################################################
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
###############################################################

# Recency skoru için
rfm["recency_score"] = pd.qcut(rfm["recency"], 5,
                               labels=[5, 4, 3, 2, 1])  # qcut fon kucukten buyuge sıralar belirli parçalara böle

# 0-100 arasındakı sayıları 5 e bol -->> 0-20, 20-40, 40-60, 60-80, 80-100 seklınde bolup kucuk ıcın 5 dıgerlerıde sıraya gore etıketler

# Monetary skoru için
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5, ])

# Frequency skoru için
rfm["frequency_score"] = pd.cut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

# Bu değerler üzerinden skor değişkeni oluşturmamız gerekiyor. R VE F bır arada olması gerekıyor M  değerini gözlemlemek için hesapladık.
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                    rfm["frequency_score"].astype(str))

rfm.describe().T

# Şampiyon müşterilerimiz kim
rfm[rfm["RFM_SCORE"] == "55"]

# Daha az değerli müşteriler için
rfm[rfm["RFM_SCORE"] == "11"]

###############################################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
###############################################################

# regex

# Segment Dic. Oluşturmak
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

# Bize yapı yakalamayı iki degeri yakalamayı saglayacak kod

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

# Oluşturulan segmentlerın analızını yapmak gerekır.

# Sınıflardaki kişilerin bilgilerine ulaşmak sitenebilir.Bunlar metrikler skorlar değil,ortalamalarını alalrak segmente gore karsılastıracagız

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# need_attention,cant loose yada At_Risk sınıfına odaklanmak istiyoruz.
rfm[rfm["segment"] == "need_attention"].head()

rfm[rfm["segment"] == "cant_loose"].head()

rfm[rfm["segment"] == "at_Risk"].head()

# Eğer bu müşterilerin ID lerine erişmek istersek.

rfm[rfm["segment"] == "new_customers"].index

rfm[rfm["segment"] == "cant_loose"].index

# Bu işlemin sonucu dışarı cıkarmak--Yeni bir df oluşurup bu id leri buraya atıyoruz.
new_df=pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# Buradaki floatları ceviriyoruz.
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# Bir excel yada csv formatında dısarıya cıkarmak gerekir.
new_df.to_csv("new_customers.csv")

rfm.to_csv("rfm.csv")

###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################

df= df_.copy()
rfm_new = create_rfm(df, csv=True)

def create_rfm(dataframe, csv=False):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
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

