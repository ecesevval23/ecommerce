# 🛒 E-Ticaret Talep Tahmini ve Akıllı Stok Yönetimi

> **Ders:** Veri Madenciliği | **Dönem:** 2024-2025 Bahar  
> **Konu:** XGBoost tabanlı zaman serisi talep tahmini ve Streamlit karar destek arayüzü

---

## 📌 Proje Hakkında

Bu proje, bir e-ticaret platformunun geçmiş satış verilerini analiz ederek **önümüzdeki 7 günün ürün talebini** tahmin eden ve bu tahminleri **aksiyon alınabilir stok kararlarına** dönüştüren bir karar destek sistemidir.

XGBoost makine öğrenmesi algoritması, 22 farklı zaman serisi özelliği ve **hibrit tahmin (hybrid forecasting)** yöntemiyle çalışır.

---

## 🏗️ Mimari ve Veri Akışı

```
synthetic_ecommerce_dataset.csv   (Ham Sipariş Kayıtları — 10.000 satır, 13 sütun)
              │
              ├─── Veri Ön İşleme (One-Hot Encoding, Normalizasyon)
              │           └──► temiz_veri.csv
              │                 ├── Anasayfa.py (KPI Paneli)
              │                 └── 2_Veri_Analizi.py (EDA)
              │
              └─── Zaman Serisi Özellik Mühendisliği
                          │   Lag (1,2,3,7,14,21,30 gün)
                          │   Rolling Mean & Std (3,7,14,30 gün)
                          ▼
               XGBoost Modeli → xgboost_demand_forecasting.pkl
                          │
                          ├── 1_Talep_Tahmini.py (7 Günlük Hibrit Tahmin)
                          └── 3_Model_Performansi.py (Metrikler)
```

---

## 📂 Dosya Yapısı

```
E-Ticaret/
├── Anasayfa.py                      # Ana dashboard — KPI metrikleri, satış trendleri
├── pages/
│   ├── 1_🔮_Talep_Tahmini.py        # 7 günlük hibrit tahmin + stok karar motoru
│   ├── 2_📊_Veri_Analizi.py         # Keşifsel veri analizi (EDA)
│   └── 3_⚙️_Model_Performansi.py    # Model metrikleri ve feature importance
├── data/
│   ├── synthetic_ecommerce_dataset.csv   # Ham veri seti
│   ├── temiz_veri.csv                    # Ön işlenmiş veri (EDA için)
│   └── xgboost_demand_forecasting.pkl    # Eğitilmiş XGBoost modeli
├── requirements.txt
└── README.md
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

### Başarım Metrikleri (Test Seti — Kronolojik %20)

| Metrik | Değer | Açıklama |
|--------|-------|---------|
| **R²** | 0.9637 | Log uzayında |
| **MAE** | 0.12 | ≈ 0.13 adet gerçek sapma |
| **RMSE** | 0.37 | ≈ 0.45 adet gerçek sapma |

### Veri Yapısı: Aralıklı Talep (Intermittent Demand)

Bir ürün **günlerin %71'inde hiç satılmıyor**, satıldığı gün ise ortalama **3.54 adet** satılıyor. Bu kalıp e-ticarette çok yaygındır. Model bu yapıyı doğru yakalıyor — tahminler 0-0-0-7-0-6 şeklinde görünür.

### Hibrit (Hybrid) Forecasting

Model, seçilen tarihe göre iki farklı modda çalışır:
1. **Geçmiş Test (Backtesting):** Seçilen tarih geçmişteyse, gerçek satış verileriyle model tahminlerini karşılaştırır.
2. **Gelecek Tahmini (Projeksiyon):** Gelecek günler için model kendi tahminlerini "gerçekleşmiş satış" gibi kabul ederek özyinelemeli şekilde haftalık projeksiyon elde eder.

---

## 🎯 Temel Tasarım Kararları

### Neden İki Farklı Veri Seti?
- `temiz_veri.csv` → One-Hot Encoded → EDA ve görselleştirme
- `synthetic_ecommerce_dataset.csv` → Ham kayıtlar → Zaman serisi özellik hesabı

### Dinamik Stok Karar Eşiği
Stok aksiyonları her ürünün kendi 30 günlük ortalamasına göre belirlenir:
- **Stok Artır:** Haftalık tahmin > Haftalık ortalama × 1.3
- **Stok Koru:** Haftalık tahmin ≥ Haftalık ortalama × 0.7
- **Stok Azalt:** Haftalık tahmin < Haftalık ortalama × 0.7

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Sanal ortamı etkinleştir
.\Scripts\activate

# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
streamlit run Anasayfa.py
```

---

## 📊 Sayfalar

| Sayfa | İşlev |
|-------|-------|
| **Anasayfa** | KPI metrikleri, aylık trend ve ödeme yöntemi grafikleri |
| **Talep Tahmini** | Ürün seçimi → 7 günlük hibrit tahmin → Haftalık stok aksiyonu → Trend grafiği |
| **Veri Analizi** | Kategori bazlı satış, günlük yoğunluk, KDE, korelasyon, scatter |
| **Model Performansı** | R²/MAE/RMSE, aralıklı talep açıklaması, feature importance |
