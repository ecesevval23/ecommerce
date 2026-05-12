# 🛒 E-Ticaret Talep Tahmini ve Akıllı Stok Yönetimi

> **Ders:** Veri Madenciliği | **Dönem:** 2025-2026 Bahar  
> **Konu:** XGBoost tabanlı zaman serisi talep tahmini ve Streamlit karar destek arayüzü

---

## 📌 Proje Hakkında

Bu proje, bir e-ticaret platformunun geçmiş satış verilerini analiz ederek **yarınki ürün talebini tahmin eden** ve bu tahmini **aksiyon alınabilir stok kararlarına** dönüştüren bir karar destek sistemidir.

Gerçek dünyada Amazon, Trendyol ve Walmart gibi şirketlerin kullandığı **Demand Forecasting (Talep Tahmin)** motorlarının temel prensiplerine dayanan bu sistem; XGBoost makine öğrenmesi algoritmasını, zaman serisi özellik mühendisliğini ve interaktif bir Streamlit arayüzünü birleştirir.

---

## 🏗️ Mimari ve Veri Akışı

```
synthetic_ecommerce_dataset.csv   (Ham Sipariş Kayıtları — 10.000 satır, 13 sütun)
              │
              ├─── Veri Ön İşleme (One-Hot Encoding, Normalizasyon)
              │           │
              │           └──► temiz_veri.csv
              │                      │
              │              ┌───────┴───────┐
              │           Anasayfa.py    2_Veri_Analizi.py
              │           (KPI Paneli)   (EDA Görselleştirme)
              │
              └─── Zaman Serisi Özellik Mühendisliği
                          │
                          │   Lag (1,2,3,7,14,21,30 gün)
                          │   Rolling Mean & Std (3,7,14,30 gün)
                          │   Tarihsel özellikler (gün, ay, hafta sonu...)
                          │
                          ▼
               XGBoost Modeli Eğitimi (Kronolojik %80/%20 split)
                          │
                          └──► xgboost_demand_forecasting.pkl
                                       │
                               1_Talep_Tahmini.py      3_Model_Performansi.py
                               (Anlık Tahmin + Karar)  (Metrikler + Açıklanabilirlik)
```

---

## 📂 Dosya Yapısı

```
E-Ticaret/
│
├── Anasayfa.py                      # Ana dashboard — KPI metrikleri, satış trendleri
│
├── pages/
│   ├── 1_🔮_Talep_Tahmini.py        # XGBoost tahmini + akıllı stok karar motoru
│   ├── 2_📊_Veri_Analizi.py         # Keşifsel veri analizi (EDA) ve görselleştirme
│   └── 3_⚙️_Model_Performansi.py    # Model metrikleri ve feature importance
│
└── data/
    ├── synthetic_ecommerce_dataset.csv   # Ham veri seti (model ve EDA için ana kaynak)
    ├── temiz_veri.csv                    # Ön işlenmiş veri (EDA görselleştirmeleri için)
    └── xgboost_demand_forecasting.pkl    # Eğitilmiş XGBoost modeli
```

---

## 🤖 Model Detayları

### Algoritma: XGBoost Regressor

| Parametre | Değer |
|-----------|-------|
| n_estimators | 1000 |
| learning_rate | 0.01 |
| max_depth | 10 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | reg:squarederror |

### Özellik Mühendisliği (22 Özellik)

| Grup | Özellikler |
|------|-----------|
| **Lag (Geçmiş Satış)** | 1, 2, 3, 7, 14, 21, 30 gün önceki satış |
| **Rolling Mean** | 3, 7, 14, 30 günlük hareketli ortalama |
| **Rolling Std** | 3, 7, 14, 30 günlük standart sapma |
| **Zamansal** | Gün, Ay, Haftanın Günü, Hafta No, Hafta Sonu, Zaman Trendi |
| **Ürün** | Label Encoded ürün kodu |

### Başarım Metrikleri (Test Seti — Kronolojik %20)

| Metrik | Değer | Açıklama |
|--------|-------|---------|
| **R² Skoru** | 0.9637 | Log uzayında hesaplanmıştır |
| **MAE** | 0.12 | Log uzayında ≈ 0.13 adet gerçek sapma |
| **RMSE** | 0.37 | Log uzayında ≈ 0.45 adet gerçek sapma |

> **Not:** Metrikler `np.log1p()` transform sonrası log uzayında hesaplanmıştır. Bu, e-ticarette yaygın olan "aralıklı talep (intermittent demand)" verisini modellemede endüstri standardı bir yaklaşımdır. Ham ölçekteki tahmin sapması ortalama **0.13 adet/gün**'dür.

---

## 🎯 Temel Tasarım Kararları

### 1. Neden İki Farklı Veri Seti?
- `temiz_veri.csv` → One-Hot Encoded hali → EDA ve görselleştirme için idealdir.
- `synthetic_ecommerce_dataset.csv` → Ham satış kayıtları → Zaman serisi özellik hesabı (`product` ve `purchase_date` sütunları) için zorunludur.
- Bu ayrım, gerçek dünya veri bilimi pipeline'larında standarttır.

### 2. Neden Sadece 1 Gün Sonrası Tahmin?
XGBoost, `lag_1` (dünkü satış) gibi anlık geçmiş verilere dayandığı için **T+1 (Ertesi Gün)** tahmini için optimize edilmiştir. Daha ileri tarihlerde model kendi ürettiği tahminleri baz almak zorunda kalır (özyineli tahmin), bu da hata oranını üstel olarak büyütür. Bu nedenle sistem tarih seçimini kaldırmış, her zaman en güvenilir tarihi (son verinin ertesi günü) otomatik hedefler.

### 3. Dinamik Stok Karar Eşiği
Stok aksiyonları sabit rakamlarla değil, **her ürünün kendi 30 günlük ortalamasına göre dinamik eşiklerle** belirlenir:
- `Yüksek Talep Eşiği = max(2, ceil(ort × 1.5))`
- `Normal Talep Eşiği = max(1, floor(ort × 0.5))`

Bu yaklaşım, günde 0.8 ortalama satan bir ürünle günde 5 ortalama satan bir ürünü aynı eşikle değerlendirme hatasını önler.

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Sanal ortamı etkinleştir (Windows)
.\Scripts\activate

# Gerekli kütüphaneleri yükle
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost

# Uygulamayı başlat
streamlit run Anasayfa.py
```

> **Not:** `requirements.txt` dosyası oluşturmak için:
> ```bash
> pip freeze > requirements.txt
> ```

---

## 📊 Sayfalar ve İşlevleri

| Sayfa | İşlev |
|-------|-------|
| **Anasayfa** | Toplam satış, ortalama fiyat, en popüler kategori KPI'ları; aylık trend ve ödeme yöntemi grafikleri |
| **Talep Tahmini** | Ürün seçimi → XGBoost ile anlık tahmin → Dinamik stok karar aksiyonu → 30 günlük trend grafiği |
| **Veri Analizi** | Kategori bazlı satış (80 ürün), haftanın günü yoğunluğu, KDE dağılımı, korelasyon matrisi, scatter analizleri |
| **Model Performansı** | R²/MAE/RMSE metrikleri, log-transform açıklaması, feature importance grafiği, model hiper-parametreleri |

---

## 🚀 GitHub'a Push Öncesi Kontrol Listesi

- [ ] `requirements.txt` oluşturuldu: `pip freeze > requirements.txt`
- [ ] `.gitignore` dosyası `data/temiz_veri.csv` ve `data/xgboost_demand_forecasting.pkl` büyük dosyalar için güncellendi (isteğe bağlı)
- [ ] Sanal ortam klasörleri (`Scripts/`, `Lib/`, `Include/`) `.gitignore`'da
- [ ] `pyvenv.cfg` `.gitignore`'da (isteğe bağlı)

> **⚠️ Dikkat:** `xgboost_demand_forecasting.pkl` dosyası ~37 MB. GitHub'un 100MB sınırı içinde ama büyük dosyalar için [Git LFS](https://git-lfs.github.com/) önerilir.
