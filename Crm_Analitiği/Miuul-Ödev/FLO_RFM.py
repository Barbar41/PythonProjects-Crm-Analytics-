
######################################
# Görev 1: Veriyi Anlama ve Hazırlama
#####################################

import pandas as pd
import datetime as dt
# Bütün Kolonları Göster
pd.set_option("display.max_columns",None)

# Bütün satırları göster
# pd.set_option("display.max_rows", None)

# Virgülden sonra iki basamak al
pd.set_option("display.float_format", lambda x:"%2.f" % x)

# Görünmeyen kolonları göster -1000 karaktere kadar
pd.set_option("display.width",100)

# Adım 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_=pd.read_csv(r"C:\Users\Baris\PycharmProjects\PythonProject2022\Crm_Analitiği\datasets\flo_data_20k.csv")
df= df_.copy()
df.head()

# Adım 2: Veri setinde
#  a. İlk 10 gözlem,
df.head(10)

#  b. Değişken isimleri,
df.columns
df.shape

#  c. Betimsel istatistik,
df.describe().T

#  d. Boş değer,
df.isnull().sum()

#  e. Değişken tipleri, incelemesi yapınız.
df.info()

#  Adım 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#  alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

##---Her bir müşterinin toplam alışveriş sayısı = online+ offline
df["order_num_total"]= df["order_num_total_ever_online"]+ df["order_num_total_ever_offline"]
df["order_num_total"].head()
##---Her bir müşterinin toplam harcaması = Offline Harcama+ Online Harcama
df["customer_value_total"]= df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df["customer_value_total"].head()

#  Adım 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

##---Değişken tiplerini gözlemleme
df.info() #--Bazı tarih değişkenleri kategorik gözüktüğü için değişken tipini değiştirmeliyiz.

##---Tarih içeren kolanlar data columns a atıldı
date_columns=df.columns[df.columns.str.contains("date")]

##---Tarih içeren bu değişkenleretopluca tip ataması yapıldı,dataframe eklendi
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()


#  Adım 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

##---Hangi alışveriş kanalında 1-Kaçar müşteri var, 2-Toplam Kaç Tane Ürün Alınmış, 3-Toplam Ne Kadar Harcama Yapılmış
df.groupby("order_channel").agg({"master_id":"count",
                                 "order_num_total":"sum",
                                 "customer_value_total":"sum"})

#  Adım 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

##--Cusotmer value total olarak olusturdugumuz dehısken online+offline
df.sort_values("customer_value_total", ascending=False)[:10]

#  Adım 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("order_num_total", ascending=False)[:10]

#  Adım 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(dataframe):
    dataframe["order_num_total"]= dataframe["order_num_total_ever_online"]+dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns= dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns]= dataframe[date_columns].apply(pd.to_datetime)
    return df

######################################
# Görev 2: RFM Metriklerinin Hesaplanması
#######################################
# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
# Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.
# Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz
# -----Recency değerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz

# Veri setindeki en son alşverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi
# En son alışveriş tarihinden 2 gün sonra analizi yaptığımızı varsayalım.Bu yüzden son tarihe 2 gün ekledik.
df["last_order_date"].max() # 2021-05-30

# datetime kütüphanesiyle analiz tarihi oluşlturulması
analysis_date= dt.datetime(2021,6,1)

# Customer_ID, Recency, Frequency ve Monetary değerleirinin yer aldığı yeni bir rfm dataframe

# Boş bir dataframe atama
rfm=pd.DataFrame()

# Müşteri Id lerin eklenmesi
rfm["customer_id"]= df["master_id"]

# Recency metriği için analiz tarihinden son sipariş tarihi çıkarılacak yeni bir değişken oluşturulması
rfm["recency"] = (analysis_date- df["last_order_date"]).astype("timedelta64[D]")#astype timedelta D ie dat tani gün cinsinden fark

# Frequency metriği müşterinin toplam alışverişi
rfm["frequency"]= df["order_num_total"]

# Müşterinin bıraktığı parasal değer
rfm["monetary"]= df["customer_value_total"]

# customer_id ve recenccy frequency monetary metriklerinin içeren dataframe kontrol
rfm.head()

##################################
# Görev 3: RF Skorunun Hesaplanması
##################################

# Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

# recency nin küçük olmasını,frequency ve monetary nin ise büyük olmasını bekleriz.

# qcut:İlgili değişkenin değerlerini küçükten büyüğe doğru sırala,çeyrekliklere göre 5 paçaya böl ve labellere göre isimlendir
rfm["recency_score"]=pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# Frekanslarda oluşabilecek sorun için rank metodu kullanıyoruz.
rfm["frequency_score"]= pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"]= pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Kontrol edelim
rfm.head()

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"]= (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
rfm["RF_SCORE"].head()
#---Recency_scıre ve frequency_score ve monetary_score tek bir değişken olarak ifade edilmesi ve RFM_SCORE olarak kayıt
rfm["RFM_SCORE"]= (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))
rfm.head()

################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
################################################
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
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

# Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
rfm["segment"]= rfm["RF_SCORE"].replace(seg_map,regex=True)

##########################
# Görev 5: Aksiyon Zamanı !
##########################

# Adım 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])


# Adım 2: RFM anali zi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

 # a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
 #   -tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak, iletişime geçmek isteniliyor.
 #   -Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
 #   -Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
#--RFm segmenti champions ve loyal customers olanların idlerini tut
target_segments_customer_ids= rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]

#---master id si target segments customer id içinde olanları ve kadın olanları cust_ids içerisinde tuttuk
cust_ids=df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]

#---cust_ids bir pandas serisi idi bunu yeni marka hedefmüşteri_id.csv olarak kaydettik
cust_ids.to_csv("yeni_marka_hedef_musteri_id.csv", index=False)

cust_ids.head()

rfm.head()

 # b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır.
 #   -Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler,
 #   -uykuda olanlar ve yeni gelen müşteriler özel olarak hedef alınmak isteniyor.
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
cust_ids= df[(df["master_id"].isin(target_segments_customer_ids))& ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]

 #   -Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.
cust_ids.to_csv("indirim_hedef_musteri_ids.csv", index=False)

###########################
###BONUS###
##########################
# Sürecin fonksiyonlaştırılması her dönem kullanılabilmesi açısından fayda sağlar.

def create_rfm(dataframe):
   # Veriyi Hazırlama
   df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
   df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
   date_columns = df.columns[df.columns.str.contains("date")]
   df[date_columns] = df[date_columns].apply(pd.to_datetime)

   # RFM Metriklerinin Hesaplanması
   df["last_order_date"].max()  # 2021-05-30
   analysis_date = dt.datetime(2021, 6, 1)
   rfm = pd.DataFrame()
   rfm["customer_id"] = df["master_id"]
   rfm["recency"] = (analysis_date - df["last_order_date"]).astype("timedelta64[D]")
   rfm["frequency"] = df["order_num_total"]
   rfm["monetary"] = df["customer_value_total"]

   # RF ve RFM Skorlarının Hesaplanması
   rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
   rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
   rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])
   rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
   rfm["RFM_SCORE"] = ( rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str) + rfm["monetary_score"].astype(str))

   # Segmentlerin İsimlendirilmesi
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

   # return:Kullanılabilir bir nesene olarak dışarı cıkarmak

   rfm_df=create_rfm(df)





