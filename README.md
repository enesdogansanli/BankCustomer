# BankCustomer
Bank Customer Data for Predicting Customer Churn

## İçindekiler

- [BankCustomer](#bankcustomer)
  - [İçindekiler](#i̇çindekiler)
  - [Özet](#özet)
  - [Giriş](#giriş)
  - [Veri Kümesi](#veri-kümesi)
  - [Sınıflandırma Modelleri](#sınıflandırma-modelleri)
    - [KNN](#knn)
    - [Naive Bayes](#naive-bayes)
    - [Decision Tree](#decision-tree)
    - [Artifical Neural Network](#artifical-neural-network)
  - [Deneysel Analiz](#deneysel-analiz)
  - [Sonuç](#sonuç)
  - [Referanslar](#referanslar)

## Özet

Bu ödev kapsamında banka müşterilerine ait veri seti üzerinde çalışılmıştır. Verilen veri seti 
banka müşterilerinin yaş, kredi notu, cinsiyet, maaş vb. bilgilerden oluşmaktadır. Veri seti 
10.000 adet örnek ve bu örneklere ait 12 adet özellikten oluşmaktadır.
Bu veri seti üzerinde önce özellik mühendisliği yapılarak verinin model eğitimi için hazır hale 
gelmesi sağlanmıştır. Daha sonra eğitim, validasyon ve test için ayrılan toplamda 1.000 adet 
veri ile KNN, Neural Network, Naive Bayes ve Decision Tree modelleri eğitilmiştir. Bu eğitim 
sırasında validasyon verisi kullanılarak her model için en iyi parametreler belirlenmiştir. Son 
olarak her model için accuracy, f1-score, precision ve recall başarı ölçme kriterleri elde edilmiş 
ve modellerin veri üzerindeki başarıları karşılaştırılmıştır. Bu başarı değerlerine göre de banka 
müşterisinin bankayı terk etme veya terk etmeme olasılığının hesaplanması hedeflenmiştir. 
Sonuçlar incelendiğinde her modeli için accuracy değeri ortalama %80 olarak gözlemlenmiştir. 
Bun karşılık f1-score, precision ve recall değerlerinin accuracy değerine göre düşük olduğu da 
gözlemlenmiştir. Bunun nedeni araştırıldığında veri setimizin dengeli bir veri seti olmadığı ve 
bu nedenle accuracy değerinin yanıltıcı olabileceği sonucuna ulaşılmıştır. Bu noktada veri 
çoğaltma teknikleri veya benzeri bir yöntem ile veri setinin dengeli hale getirilmesi bir çözüm 
olarak değerlendirilebilir.

## Giriş

Veri günümüzün yeni madeni olarak adlandırılmaktadır. Elinde veri bulunduran firmalarda bu 
madeni en iyi şekilde kullanmanın yollarını aramaya başlamışlardır. Bu firmaların içinde de 
çoğunluğu bankalarının oluşturduğu söylenebilir. Çünkü yapılan iş gereği birçok kişinin 
verisini bünyesinde barındırması gerekmektedir. Bankalararası Kart Merkezinin yapmış olduğu 
bir araştırmada her 5 yetişkinden 2’sinin kredi kartı sahibi olması bankaların ne kadar çok veri 
sahibi olduğunu doğrular niteliktedir.

Bu çalışmada veri setimiz banka müşterilerine ait özellikler ve müşterinin bankayı terk edip 
terk etmediği bilgisini içermektedir. Bankalar için hayati öneme sahip olan müşterilerin 
kaybının önüne geçilmesi çoğu zaman yeni müşteri kazanmaktan daha az maliyet gerektirir. Bu 
da bankları müşteri kaybının önüne geçmeyi yönelik yöntemler aramaya sevk etmiştir. Kısaca
bu çalışmada amaç müşterinin henüz bankayı terk etmeden tespit edilip, o müşterinin bankada 
kalmasını sağlayacak çeşitli adımların önceden atılması olarak söylenebilir.

Veriden anlamlı bilgi elde etme söz konusu olduğunda devreye makine öğrenmesi yöntemleri 
girmektedir. Çünkü günümüzde veri o kadar yüksek boyutlara ulaşmıştır ki insanların bu 
verileri elle analiz ederek anlamlı sonuçlar çıkarabilmesi imkânsız hale gelmiştir. Aynı 
zamanda sonuçların doğruluğu, zaman ve performans gibi kriterler baz alındığında makine 
öğrenmesi yöntemleri çok daha fazla öne çıkmaktadır. Özellikle son zamanlarda veri biliminde 
yaşanan gelişmelere ek olarak teknolojik cihazlarında güçlenmesi ile büyük veriler üzerinde 
bile kısa sürede başarılı sonuçlar elde edilmiştir. Neredeyse her sektörün bünyesinde veri barındırmasıyla birlikte makine öğrenmesi yöntemlerinin de her sektör için kullanılabilir 
olmanın önünü açılmıştır.

## Veri Kümesi

Veri setimiz banka müşterilerine ait bilgileri içermektedir. Toplamda 10.000 adet örnek ve 12 
adet özellikten oluşmaktadır. Özelliklerden 11 tanesi kişiye ait bilgileri içerirken, bir özellik o 
kişinin bankayı terk edip terk etmediği bilgisini tutmaktadır.

1. **Customer_id:** Müşteri Kimlik Numarası (Tam sayı)
2. **Credit_score:** Müşteri Kredi Notu (Tam sayı)
3. **Country:** Müşterinin Bulunduğu Ülke (String) (France, Spain, Germany)
4. **Gender:** Müşterinin Cinsiyeti (String) (Male, Female)
5. **Age:** Müşterinin Yaşı (Tam sayı) (En küçük=18, Ortalama=38.92, En büyük=92)
6. **Tenure:** Müşterinin Kaç Yıllık Kullanım Süresi (Tam sayı) (En az=0, Ortalama=5.012, En fazla=10)
7. **Balance:** Müşteri Hesap Dengesi (Kesirli sayı) (En az=0, Ortalama=76485.889, En fazla=250898.09)
8. **Products_number:** Müşteri Bankada Kullandığı Ürün Sayısı (Tam sayı) (En az=0, Ortalama=1.5302, En fazla=4)
9. **Credit_card:** Müşterinin Sahip Olduğu Kredi Kartı Sayısı (Tam sayı) (En az=0,  Ortalama=0.705, En fazla=1)
10. **Active_member:** Müşterinin Aktiflik Durumu (Tam sayı) (0, 1)
11. **Estimated_salary:** Tahmini Müşteri Maaşı (Kesirli sayı) (En az=11.58,  Ortalama=100090.239, En fazla=199992.48)
12. **Churn :** Müşterinin Bankayı Terk Etme Durumu (Tam sayı) (0, 1)

Veri seti toplamda 10.000 adet örnek içerse de bu örnekler içerisinden rastgele seçilen toplamda 
1.000 adet örnek ile çalışmıştır. Bu 1.000 adet örnek içerisinden rastgele seçilen 400 tanesi 
eğitim için, rastgele seçilen 300 tanesi validasyon için ve geri kalan 300 tane örnekte test için 
kullanılmıştır. Eğitim verisi ile model eğitimi gerçekleştirilmiştir. Validasyon verisi ile model 
için en uygun parametrelerin belirlenmesi işlemi gerçekleştirilmiştir. Son olarak test verisi ile 
modelimizin başarısı gözlemlenmiştir.

## Sınıflandırma Modelleri

Bu ödevde sınıflandırma modelleri olarak KNN (K-Nearest Neighbors), Naive Bayes, Decision 
Tree ve Artifical Neural Network (Backpropagation) kullanılmıştır. Her modelin kendine ait 
bir çalışma mantığı ve kriteri mevcuttur.

### KNN

K-en yakın komşuluk belirtilen k adet en yakın komşuyu dikkate alarak ilgili örneğin 
sınıflandırma işlemini yapan bir model olarak ifade edilebilir. İki temel değer olan komşuluk 
sayısı ve mesafe üzerinden tahminlerini gerçekleştirir. Hesaplamaya dayalı bir yöntem olduğu 
için model eğitiminden önce verinin içerisindeki karakter içeren özellikler sayısal değerlere 
dönüştürülmelidir.

### Naive Bayes

Naive bayes, bayes teoreminden türetilmiş ve koşullu olasılık temelline dayanan bir öğrenme 
modelidir. Özelliklerinden birbirinden bağımsız olduğu kabulüne göre işlem yaptığı için naive 
bayes olarak adlandırılmaktadır. Özellikle spam-mail sınıflandırma gibi işlemlerde yüksek 
başarı göstermektedir. Hem sayısal hem de kategorik verilerde kullanılabilir.

### Decision Tree

Karar ağaçları en çok kullanılan gözetimli öğrenme modellerinden birisidir. Veri seti üzerinde 
bir dizi karar kuralları uygulayarak daha küçük dallar oluşturmakta ve bu yapı üzerinden karar 
verme sürecini gerçekleştirmektedir. Oldukça kullanışlı ve anlaşılır bir yöntemdir. Hem sayısal 
hem de kategorik verilerde kullanılabilir.

### Artifical Neural Network

Yapay nöron ağları genel olarak 3 temel katmandan (input layer, hidden layer, output layer) 
oluşan ve düğüm adı verilen yapıların ağırlıklandırılarak matematiksel işlem üzerinden sonuç 
elde etmemize yarayan öğrenme modelleridir. Modelin karmaşıklığına göre hidden layer 
eklenmesi ve düğüm sayısı belirleme işlemi yapılabilir. Bir adet input layer ve bir adet output 
layer her model için sabittir. Matematiksel işlem tabanlı bir model olduğu için model eğitimine 
verilecek verinin kategorik değişkenlerinin sayısal değişkene dönüştürülmesi gerekir.

## Deneysel Analiz

* Veri Ön İşleme
* Model Eğitimi
* Doğrulama Aşaması
* Test Aşaması

## Sonuç

Sonuç olarak veri seti üzerinde özellik mühendisliği başarılı bir şekilde uygulanmış, veri 
içerisinden gerekli bilgiler elde edilmiş ve veri eğitim için hazır hale getirilmiştir. Daha sonra 
modellerin hiper parametre analizleri yapılarak her model için en iyi parametreler belirlenmiş 
bu parametreleri ile en iyi modeller oluşturulmuştur.
Model başarıları incelendiğinde accuracy değerinin en yüksek Neural Network modelinde 
%81,6 olarak elde edildiği gözlemlenmiştir. Sadece accuracy değeri baz alındığında güzel bir 
başarı elde edilmiştir. Diğer modellerde de bu sonuca yakın accuracy değeri elde edildiği 
gözlemlenmiştir. Buna karşılık f1-score değeri accuracy değerine göre oldukça düşük çıkmıştır. 
Böyle bir durumun oluşmasının asıl nedeninin dengeli olmayan bir veri seti ile çalışması olduğu 
düşünülmektedir. Veri seti bir şekilde dengeli hale getirilse f1-score değerinin yüksek bir değer 
olacağı söylenebilir.

## Referanslar

* [Kart Sayıları | Bankalararası Kart Merkezi (bkm.com.tr)](https://bkm.com.tr/kart-sayilari/)
* [sklearn.metrics.confusion_matrix — scikit-learn 1.2.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* [Data](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)