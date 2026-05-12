import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Akıllı Talep Tahmini & Stok Yönetimi", layout="wide")

st.title("🔮 Akıllı Talep Tahmini ve Karar Destek Sistemi")
st.markdown("Yapay zeka modelimiz seçtiğiniz ürünün **son 30 günlük geçmişini** analiz ederek önümüzdeki **7 günlük talebi** tahmin eder ve aksiyon alınabilir stok tavsiyeleri sunar.")

@st.cache_resource
def load_model_and_data():
    try:
        csv_path = os.path.join("data", "synthetic_ecommerce_dataset.csv")
        model_path = os.path.join("data", "xgboost_demand_forecasting.pkl")
        df = pd.read_csv(csv_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return df, model
    except Exception as e:
        return None, str(e)

@st.cache_data
def build_daily_sales(df):
    """Eğitim kodunun birebir pipeline'ını uygular."""
    df_copy = df.copy()
    df_copy['purchase_date'] = pd.to_datetime(df_copy['purchase_date'])
    df_copy = df_copy.sort_values('purchase_date')
    
    start_date = df_copy['purchase_date'].min()
    end_date = df_copy['purchase_date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_products = df_copy['product'].unique()
    
    full_index = pd.MultiIndex.from_product([all_dates, all_products], names=['purchase_date', 'product'])
    daily_sales = df_copy.groupby(['purchase_date', 'product'])['quantity'].sum().reindex(full_index, fill_value=0).reset_index()
    
    daily_sales['day'] = daily_sales['purchase_date'].dt.day
    daily_sales['month'] = daily_sales['purchase_date'].dt.month
    daily_sales['weekday'] = daily_sales['purchase_date'].dt.weekday
    daily_sales['weekofyear'] = daily_sales['purchase_date'].dt.isocalendar().week.astype(int)
    daily_sales['is_weekend'] = (daily_sales['weekday'] >= 5).astype(int)
    daily_sales['trend'] = np.arange(len(daily_sales))
    
    le = LabelEncoder()
    daily_sales['product_encoded'] = le.fit_transform(daily_sales['product'])
    
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        daily_sales[f'lag_{lag}'] = daily_sales.groupby('product')['quantity'].shift(lag)
    for window in [3, 7, 14, 30]:
        daily_sales[f'rolling_mean_{window}'] = daily_sales.groupby('product')['quantity'].transform(lambda x: x.rolling(window).mean())
        daily_sales[f'rolling_std_{window}'] = daily_sales.groupby('product')['quantity'].transform(lambda x: x.rolling(window).std())
    
    daily_sales = daily_sales.dropna()
    return daily_sales

FEATURE_COLS = [
    'product_encoded', 'day', 'month', 'weekday', 'weekofyear',
    'is_weekend', 'trend', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
    'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_3', 'rolling_std_7', 'rolling_std_14', 'rolling_std_30'
]

def recursive_forecast(model, daily_sales, product, n_days=7):
    """
    Recursive (Özyinelemeli) Tahmin:
    Her gün bir önceki tahmini gerçekmiş gibi kabul ederek N gün ileriye tahmin yapar.
    """
    product_data = daily_sales[daily_sales['product'] == product].copy()
    if product_data.empty:
        return None
    
    # Son satırı baz al
    current_row = product_data.iloc[-1].copy()
    son_tarih = current_row['purchase_date']
    
    # Geçmiş satış geçmişini (rolling hesabı için) listeye al
    recent_sales = list(product_data['quantity'].values[-30:])
    
    predictions = []
    
    for step in range(1, n_days + 1):
        target_date = son_tarih + pd.Timedelta(days=step)
        
        # Yeni satır oluştur
        new_row = current_row.copy()
        new_row['purchase_date'] = target_date
        new_row['day'] = target_date.day
        new_row['month'] = target_date.month
        new_row['weekday'] = target_date.weekday()
        new_row['weekofyear'] = int(target_date.isocalendar().week)
        new_row['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
        new_row['trend'] = current_row['trend'] + step
        
        # Lag'ları son satışlardan hesapla
        n = len(recent_sales)
        new_row['lag_1'] = recent_sales[-1] if n >= 1 else 0
        new_row['lag_2'] = recent_sales[-2] if n >= 2 else 0
        new_row['lag_3'] = recent_sales[-3] if n >= 3 else 0
        new_row['lag_7'] = recent_sales[-7] if n >= 7 else 0
        new_row['lag_14'] = recent_sales[-14] if n >= 14 else 0
        new_row['lag_21'] = recent_sales[-21] if n >= 21 else 0
        new_row['lag_30'] = recent_sales[-30] if n >= 30 else 0
        
        # Rolling hesapları
        for w in [3, 7, 14, 30]:
            window_data = recent_sales[-w:] if n >= w else recent_sales
            new_row[f'rolling_mean_{w}'] = np.mean(window_data)
            new_row[f'rolling_std_{w}'] = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
        
        # Tahmin
        X = pd.DataFrame([new_row])[FEATURE_COLS]
        pred_log = model.predict(X)[0]
        pred_real = float(np.clip(np.expm1(pred_log), 0, None))
        pred_int = round(pred_real)  # Tam sayıya yuvarla
        
        predictions.append({
            'Tarih': target_date,
            'Gün': target_date.strftime('%A'),
            'Tahmin': pred_int
        })
        
        # Bu tahmini "gerçekmiş gibi" geçmişe ekle (Recursive adım)
        recent_sales.append(pred_real)
    
    return pd.DataFrame(predictions)

# ─── SAYFA BAŞLANGICI ─────────────────────────────────────────────────────────
res = load_model_and_data()
if isinstance(res, tuple) and res[0] is not None:
    df, model = res
else:
    st.error(f"🚨 Dosyalar yüklenirken hata oluştu: {res}")
    st.stop()

kategoriler = sorted(df['category'].unique())

st.divider()

col1, col2 = st.columns(2)
with col1:
    secilen_kat = st.selectbox("🏷️ Kategori (Filtreleme)", kategoriler)
with col2:
    kategori_urunleri = df[df['category'] == secilen_kat]['product'].unique()
    secilen_urun = st.selectbox("📦 Tahmin Edilecek Ürün", sorted(kategori_urunleri))

son_tarih_pd = pd.to_datetime(df['purchase_date']).max()

# Gün Türkçe isimleri
gun_tr = {"Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
          "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"}

st.info(f"📅 **Veri Setindeki Son Tarih:** {son_tarih_pd.strftime('%d.%m.%Y')} — Model, bu tarihten sonraki **7 günün** talebini özyinelemeli (recursive) yöntemle tahmin eder.")

if st.button("🚀 7 Günlük Akıllı Analizi Başlat", use_container_width=True, type="primary"):
    with st.spinner("Model 7 günlük recursive tahmin yapıyor... Her gün bir öncekini baz alır."):
        daily_sales = build_daily_sales(df)
        
        tahmin_df = recursive_forecast(model, daily_sales, secilen_urun, n_days=7)
        
        if tahmin_df is None or tahmin_df.empty:
            st.error("Seçilen ürün için hesaplama yapılamadı.")
        else:
            # Geçmiş veriler
            gecmis = daily_sales[daily_sales['product'] == secilen_urun]
            son_30_gun = gecmis.tail(30)
            ort_satis = son_30_gun['quantity'].mean() if not son_30_gun.empty else 1.0
            if pd.isna(ort_satis) or ort_satis == 0:
                ort_satis = 0.5
            
            # 7 günlük toplam
            toplam_7gun = tahmin_df['Tahmin'].sum()
            haftalik_ort = ort_satis * 7  # Geçmiş haftalık beklenti
            
            # Stok kararı (haftalık bazda)
            if toplam_7gun >= haftalik_ort * 1.3:
                stok_karari = "🟢 Stok Artır"
                detayli_karar = f"7 günlük toplam tahmin (**{toplam_7gun} adet**), geçmiş haftalık ortalamanın (**{haftalik_ort:.0f} adet**) üstünde. Stokları artırın!"
                mesaj_tipi = "success"
            elif toplam_7gun >= haftalik_ort * 0.7:
                stok_karari = "🟡 Stok Koru"
                detayli_karar = f"7 günlük toplam tahmin (**{toplam_7gun} adet**) geçmiş haftalık ortalamayla (**{haftalik_ort:.0f} adet**) uyumlu. Mevcut stok düzeyini koruyun."
                mesaj_tipi = "warning"
            else:
                stok_karari = "🔴 Stok Azalt"
                detayli_karar = f"7 günlük toplam tahmin (**{toplam_7gun} adet**), geçmiş haftalık ortalamanın (**{haftalik_ort:.0f} adet**) altında. Talep durgunluğu bekleniyor."
                mesaj_tipi = "error"
            
            # ─── ÜST METRİKLER ───────────────────────────────────────
            st.divider()
            st.subheader(f"📊 {secilen_urun} — 7 Günlük Tahmin Sonucu")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(label="📦 7 Günlük Toplam Tahmin", value=f"{toplam_7gun} Adet")
            c2.metric(label="📊 Günlük Ortalama", value=f"{toplam_7gun / 7:.1f} Adet/Gün")
            c3.metric(label="⚙️ Stok Aksiyonu", value=stok_karari)
            c4.metric(label="📈 Geçmiş Haftalık Ort.", value=f"{haftalik_ort:.0f} Adet",
                      delta=f"{toplam_7gun - haftalik_ort:+.0f} fark",
                      help="Son 30 günlük ortalamanın 7 günlük karşılığı")
            
            # Aksiyon mesajı
            if mesaj_tipi == "success":
                st.success(f"**Aksiyon Planı:** {detayli_karar}")
            elif mesaj_tipi == "warning":
                st.warning(f"**Aksiyon Planı:** {detayli_karar}")
            else:
                st.error(f"**Aksiyon Planı:** {detayli_karar}")
            
            # ─── GÜNLÜK TAHMİN TABLOSU ───────────────────────────────
            st.markdown("#### 📋 Günlük Tahmin Detayı")
            
            tablo = tahmin_df.copy()
            tablo['Tarih'] = tablo['Tarih'].dt.strftime('%d.%m.%Y')
            tablo['Gün'] = tablo['Gün'].map(gun_tr)
            tablo['Tahmin'] = tablo['Tahmin'].apply(lambda x: f"{x} Adet")
            tablo.index = range(1, len(tablo) + 1)
            tablo.index.name = "Gün #"
            st.dataframe(tablo, use_container_width=True)
            
            # ─── GEÇMİŞ BAĞLAM ──────────────────────────────────────
            st.markdown("#### 📆 Geçmiş Performans Bağlamı")
            mc1, mc2, mc3 = st.columns(3)
            
            son_7_gun_gecmis = gecmis.tail(7)
            toplam_30 = int(son_30_gun['quantity'].sum())
            ort_7g = round(son_7_gun_gecmis['quantity'].mean(), 1) if not son_7_gun_gecmis.empty else 0
            
            mc1.metric(label="📅 Son 30 Gün Toplam", value=f"{toplam_30} Adet")
            mc2.metric(label="📊 Son 7 Gün Ort.", value=f"{ort_7g} Adet/Gün")
            mc3.metric(label="📌 Son 30 Gün Ort.", value=f"{round(ort_satis, 1)} Adet/Gün")
            
            # ─── GRAFİK: Geçmiş 30 Gün + Gelecek 7 Gün ─────────────
            st.markdown("### 📉 30 Günlük Geçmiş + 7 Günlük Projeksiyon")
            
            fig, ax = plt.subplots(figsize=(14, 5))
            
            # Geçmiş
            ax.plot(son_30_gun['purchase_date'], son_30_gun['quantity'],
                    marker='o', markersize=4, linestyle='-', linewidth=2,
                    color='#2980b9', label='Geçmiş Satışlar', zorder=3)
            ax.fill_between(son_30_gun['purchase_date'], son_30_gun['quantity'],
                            alpha=0.1, color='#2980b9')
            
            # Gelecek tahminler
            tahmin_tarihleri = tahmin_df['Tarih']
            tahmin_degerleri = tahmin_df['Tahmin']
            
            # Son gerçek günden ilk tahmine bağlantı çizgisi
            if not son_30_gun.empty:
                son_gercek_tarih = son_30_gun.iloc[-1]['purchase_date']
                son_gercek_satis = son_30_gun.iloc[-1]['quantity']
                ax.plot([son_gercek_tarih, tahmin_tarihleri.iloc[0]],
                        [son_gercek_satis, tahmin_degerleri.iloc[0]],
                        linestyle='--', linewidth=1.5, color='#e74c3c', alpha=0.5)
            
            ax.plot(tahmin_tarihleri, tahmin_degerleri,
                    marker='*', markersize=12, linestyle='-', linewidth=2,
                    color='#e74c3c', label='7 Günlük Tahmin', zorder=4)
            ax.fill_between(tahmin_tarihleri, tahmin_degerleri,
                            alpha=0.1, color='#e74c3c')
            
            # Tahmin noktalarına değer yaz
            for tarih, deger in zip(tahmin_tarihleri, tahmin_degerleri):
                if deger > 0:
                    ax.annotate(f"{deger}", xy=(tarih, deger),
                                xytext=(0, 8), textcoords='offset points',
                                color='#e74c3c', fontweight='bold', fontsize=8, ha='center')
            
            ax.set_ylabel("Satış Adedi", fontsize=10)
            ax.tick_params(axis='x', rotation=35, labelsize=8)
            ax.tick_params(axis='y', labelsize=9)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.grid(True, axis='y', alpha=0.2, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
            
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ─── Model Kapsamı Notu ──────────────────────────────────
            with st.expander("ℹ️ Recursive Forecasting Hakkında"):
                st.markdown("""
**Özyinelemeli (Recursive) Tahmin Nedir?**

Model tek seferde sadece 1 gün ileriye tahmin yapabilir. Ancak bu tahmini "gerçekleşmiş satış" gibi kabul edip bir sonraki günü de tahmin edebiliriz. Bu döngüyü tekrarlayarak 7 günlük projeksiyon elde ederiz.

**⚠️ Dikkat:** Her adımda hata payı birikir. İlk günlerin tahmini daha güvenilir, son günler daha belirsizdir.

| Gün | Güvenilirlik |
|-----|-------------|
| T+1 | ⭐⭐⭐⭐⭐ Çok Yüksek |
| T+2-3 | ⭐⭐⭐⭐ Yüksek |
| T+4-5 | ⭐⭐⭐ Orta |
| T+6-7 | ⭐⭐ Kabul Edilebilir |

**📍 Gelişim Yol Haritası:**
- **Aylık Tahmin:** Facebook Prophet veya LSTM ile mevsimsellik analizi
- **ERP Entegrasyonu:** Gerçek depo seviyeleriyle bağlantı kurularak otomatik sipariş üretimi
                """)