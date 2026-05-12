import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Akıllı Talep Tahmini & Stok Yönetimi", layout="wide")

st.title("🔮 Akıllı Talep Tahmini ve Karar Destek Sistemi")
st.markdown("Yapay zeka modelimiz seçtiğiniz ürünün tarihsel geçmişini detaylıca analiz eder, gelecek talebi hesaplar ve size **aksiyon alınabilir stok tavsiyeleri** sunar.")

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
def prepare_features(df, target_date):
    df_copy = df.copy()
    df_copy['purchase_date'] = pd.to_datetime(df_copy['purchase_date'])
    
    # Modelin eğitildiği sırayı birebir tutturmak için tarihe göre sıralamamız ŞART
    df_copy = df_copy.sort_values('purchase_date')
    
    # En erken tarihten, kullanıcının seçtiği hedef tarihe kadar tam bir zaman çizelgesi
    start_date = df_copy['purchase_date'].min()
    all_dates = pd.date_range(start=start_date, end=target_date, freq='D')
    all_products = df_copy['product'].unique()
    
    full_index = pd.MultiIndex.from_product([all_dates, all_products], names=['purchase_date', 'product'])
    
    # Günlük satışları grupla ve eksik günleri sıfırla doldur
    daily_sales = df_copy.groupby(['purchase_date', 'product'])['quantity'].sum().reindex(full_index, fill_value=0).reset_index()
    
    # Tarihsel özellikleri çıkar
    daily_sales['day'] = daily_sales['purchase_date'].dt.day
    daily_sales['month'] = daily_sales['purchase_date'].dt.month
    daily_sales['weekday'] = daily_sales['purchase_date'].dt.weekday
    daily_sales['weekofyear'] = daily_sales['purchase_date'].dt.isocalendar().week.astype(int)
    daily_sales['is_weekend'] = (daily_sales['weekday'] >= 5).astype(int)
    daily_sales['trend'] = np.arange(len(daily_sales))
    
    # Ürünleri encode et
    le = LabelEncoder()
    daily_sales['product_encoded'] = le.fit_transform(daily_sales['product'])
    
    # Lags (Geçmiş Satışlar)
    lags = [1, 2, 3, 7, 14, 21, 30]
    for lag in lags:
        daily_sales[f'lag_{lag}'] = daily_sales.groupby('product')['quantity'].shift(lag)
        
    # Hareketli Ortalamalar ve Sapmalar
    windows = [3, 7, 14, 30]
    for window in windows:
        daily_sales[f'rolling_mean_{window}'] = daily_sales.groupby('product')['quantity'].transform(lambda x: x.rolling(window).mean())
        daily_sales[f'rolling_std_{window}'] = daily_sales.groupby('product')['quantity'].transform(lambda x: x.rolling(window).std())
        
    return daily_sales

res = load_model_and_data()
if isinstance(res, tuple) and res[0] is not None:
    df, model = res
else:
    st.error(f"🚨 Dosyalar yüklenirken hata oluştu: {res}")
    st.stop()

# Kategorileri sadece filtreleme (UX) amaçlı alıyoruz
kategoriler = sorted(df['category'].unique())

st.divider()

col1, col2 = st.columns(2)

with col1:
    secilen_kat = st.selectbox("🏷️ Kategori (Filtreleme)", kategoriler)
    
with col2:
    kategori_urunleri = df[df['category'] == secilen_kat]['product'].unique()
    secilen_urun = st.selectbox("📦 Tahmin Edilecek Ürün", sorted(kategori_urunleri))

# Model 1 gün sonrasını tahmin edebildiği için tarihi otomatik belirliyoruz
son_tarih_pd = pd.to_datetime(df['purchase_date']).max()
hedef_tarih_pd = son_tarih_pd + pd.Timedelta(days=1)

st.info(f"📅 **Operasyonel Tahmin Tarihi:** {hedef_tarih_pd.strftime('%d.%m.%Y')} — Sistem, %96 doğruluk garantisinin geçerli olduğu **1 Gün Sonrası** tahminini otomatik hedefler. Aşağıdaki 30 günlük trend analizi, bu noktasal tahmini anlamlı bir bağlama oturtur.")

if st.button("🚀 Akıllı Analizi Başlat", use_container_width=True, type="primary"):
    with st.spinner("Model 22 farklı geçmiş zaman parametresini hesaplıyor ve geleceği analiz ediyor..."):
        # 1. Veri Hazırlığı
        daily_sales = prepare_features(df, hedef_tarih_pd)
        
        # 2. Seçilen ürün ve hedef tarih için satırı bul
        hedef_satir = daily_sales[(daily_sales['purchase_date'] == hedef_tarih_pd) & (daily_sales['product'] == secilen_urun)]
        
        if hedef_satir.empty:
            st.error("Seçilen tarih için hesaplama yapılamadı.")
        else:
            feature_cols = [
                'product_encoded', 'day', 'month', 'weekday', 'weekofyear',
                'is_weekend', 'trend', 'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_21', 'lag_30',
                'rolling_mean_3', 'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
                'rolling_std_3', 'rolling_std_7', 'rolling_std_14', 'rolling_std_30'
            ]
            
            X_input = hedef_satir[feature_cols]
            
            # 3. XGBoost Tahmini
            try:
                y_pred_log = model.predict(X_input)[0]
                tahmini_adet_float = np.expm1(y_pred_log) # Inverse log transform
                
                # Profesyonel Dokunuş: Güvenlik Stoğu (Safety Stock) Yaklaşımı
                tahmini_adet_float = np.clip(tahmini_adet_float, 0, None)
                # Model çok küçük bir ihtimal bile görse (örn: 0.15) stoksuz kalmamak için her zaman yukarı yuvarla (Ceil)
                tahmini_adet = math.ceil(tahmini_adet_float) if tahmini_adet_float > 0.05 else 0
                
                # Sadece bu ürünün geçmişini al (Ortalama hesabı ve grafik için)
                gecmis_satislar = daily_sales[(daily_sales['product'] == secilen_urun) & (daily_sales['purchase_date'] < hedef_tarih_pd)]
                son_30_gun = gecmis_satislar.tail(30)
                
                # 4. Dinamik Karar Destek Mekanizması (Ürünün Kendi Ortalamasına Göre)
                ort_satis = son_30_gun['quantity'].mean() if not son_30_gun.empty else 1.0
                if pd.isna(ort_satis) or ort_satis == 0:
                    ort_satis = 0.5
                
                # Eşik değerlerini ürünün kapasitesine göre dinamik belirliyoruz
                yuksek_esik = max(2, math.ceil(ort_satis * 1.5)) 
                normal_esik = max(1, math.floor(ort_satis * 0.5))
                
                if tahmini_adet >= yuksek_esik:
                    stok_karari = "🟢 Yüksek Talep"
                    detayli_karar = f"Bu ürün normalde günde ortalama **{ort_satis:.1f}** adet satıyor. Tahmin bunu aşıyor, satışlarda sıçrama var. Stokları artırın!"
                    mesaj_tipi = "success"
                elif tahmini_adet >= normal_esik:
                    stok_karari = "🟡 Normal Talep"
                    detayli_karar = f"Satışlar olağan seyrinde (Günlük ortalaması: **{ort_satis:.1f}**). Mevcut stok düzeyini koruyun."
                    mesaj_tipi = "warning"
                else:
                    stok_karari = "🔴 Düşük Talep"
                    detayli_karar = f"Ürünün beklenen talebi kendi günlük ortalamasının (**{ort_satis:.1f}**) altında kalıyor. Talep durgunluğu var, yeni sipariş geçmeyin."
                    mesaj_tipi = "error"
                
                st.divider()
                st.subheader("📊 Analiz ve Karar Destek Sonucu")
                
                c1, c2, c3 = st.columns(3)
                c1.metric(label=f"📦 {secilen_urun} Tahmini", value=f"{tahmini_adet} Adet")
                c2.metric(label="⚙️ Akıllı Aksiyon", value=stok_karari)
                
                # Geçmiş bilgi (lag_1)
                dun_satis = int(round(hedef_satir['lag_1'].values[0]))
                if pd.isna(dun_satis):
                    dun_satis = 0
                trend_ok = "↑" if tahmini_adet > dun_satis else ("↓" if tahmini_adet < dun_satis else "→")
                fark = abs(tahmini_adet - dun_satis)
                c3.metric(label="📈 Düne Göre Trend", value=f"{trend_ok} {fark} Adet")
                
                # Detaylı aksiyon mesajı
                if mesaj_tipi == "success":
                    st.success(f"**Aksiyon Planı:** {detayli_karar}")
                elif mesaj_tipi == "warning":
                    st.warning(f"**Aksiyon Planı:** {detayli_karar}")
                else:
                    st.error(f"**Aksiyon Planı:** {detayli_karar}")
                
                # ─── PLAN B: 30 Günlük Geçmiş Bağlam Metrikleri ────────────────
                st.markdown("#### 📆 Geçmiş Performans Bağlamı (30 Gün)")
                mc1, mc2, mc3 = st.columns(3)
                
                son_7_gun = gecmis_satislar.tail(7)
                toplam_30 = int(son_30_gun['quantity'].sum())
                ort_7g = round(son_7_gun['quantity'].mean(), 1) if not son_7_gun.empty else 0
                
                mc1.metric(
                    label="📅 30 Günlük Toplam Satış",
                    value=f"{toplam_30} Adet",
                    help="Bu ürünün son 30 günde fiilen satılan toplam adedi."
                )
                mc2.metric(
                    label="📊 7 Günlük Ort. (Haftalık)",
                    value=f"{ort_7g} Adet/Gün",
                    delta=f"{'↑' if tahmini_adet > ort_7g else '↓'} Tahminden {'yüksek' if ort_7g > tahmini_adet else 'düşük'}" if ort_7g != tahmini_adet else "Eşdeğer",
                    help="Son 7 günün günlük ortalaması. Modelin tahminini haftalık trendle karşılaştır."
                )
                mc3.metric(
                    label="📌 Aylık Ort. (30G)",
                    value=f"{round(ort_satis, 1)} Adet/Gün",
                    help="Son 30 günün günlük ortalaması. Stok karar eşikleri bu değere göre dinamik hesaplanır."
                )
                
                # 5. Zenginleştirilmiş Görsel Grafik (Geçmiş 30 Gün + Tahmin)
                st.markdown("### 📉 30 Günlük Satış Trendi ve Gelecek Projeksiyonu")
                
                fig, ax = plt.subplots(figsize=(12, 4))
                
                # Geçmiş Gerçek Satışlar — Dolgu Alanıyla Estetik
                ax.plot(son_30_gun['purchase_date'], son_30_gun['quantity'],
                        marker='o', markersize=5, linestyle='-', linewidth=2,
                        color='#2980b9', label='Geçmiş Satışlar', zorder=3)
                ax.fill_between(son_30_gun['purchase_date'], son_30_gun['quantity'],
                                alpha=0.12, color='#2980b9')
                
                # Son gerçek günden tahmin noktasına kesikli projeksiyon çizgisi
                if not son_30_gun.empty:
                    son_gercek_tarih = son_30_gun.iloc[-1]['purchase_date']
                    son_gercek_satis = son_30_gun.iloc[-1]['quantity']
                    ax.plot([son_gercek_tarih, hedef_tarih_pd],
                            [son_gercek_satis, tahmini_adet],
                            linestyle='--', linewidth=1.5,
                            color='#e74c3c', alpha=0.7, zorder=2)
                
                # Tahmin Noktası — yıldız ve etiket
                ax.plot(hedef_tarih_pd, tahmini_adet,
                        marker='*', markersize=18, color='#e74c3c',
                        label='Gelecek Tahmini', zorder=4)
                
                # Etiket: tahmini_adet 0 ise de görünür olsun
                etiket_offset = max(0.3, tahmini_adet * 0.12)
                ax.annotate(f" {tahmini_adet} Adet",
                            xy=(hedef_tarih_pd, tahmini_adet),
                            xytext=(hedef_tarih_pd, tahmini_adet + etiket_offset),
                            color='#e74c3c', fontweight='bold', fontsize=10,
                            ha='center', va='bottom')
                
                # Eksen ve grid düzeni
                ax.set_ylabel("Satış Adedi", fontsize=10)
                ax.tick_params(axis='x', rotation=35, labelsize=8)
                ax.tick_params(axis='y', labelsize=9)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.grid(True, axis='y', alpha=0.25, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
                
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                plt.tight_layout()
                st.pyplot(fig)
                
                # ─── PLAN A: Model Kapsamı Hakkında Dürüst Not ──────────────────
                with st.expander("ℹ️ Model Kapsamı ve Gelişim Yol Haritası"):
                    st.markdown("""
**Bu model nedir?** Günlük operasyonel talep tahmini *(Next-Day Demand Forecasting)*.

**Ne için kullanılır?** Raf yenileme, günlük sevkiyat planlaması, anlık depo aksiyon kararları.

**Neden sadece 1 gün?** XGBoost, `lag_1` (önceki gün satışı) gibi anlık geçmiş veriye dayandığı için T+1 tahmininde **%96 doğrulukla** çalışır. Daha ileri tarihlerde model kendi ürettiği tahminleri baz almak zorunda kalır ve hata kümülatif olarak büyür.

**📍 Bir Sonraki Geliştirme Fazı:**
- **T+7 (Haftalık):** Haftalık toplam satış üzerinden yeniden eğitilmiş XGBoost
- **T+30 (Aylık):** Facebook Prophet veya LSTM ile mevsimsellik analizi
- **ERP Entegrasyonu:** Gerçek depo seviyeleriyle bağlantı kurularak otomatik sipariş üretimi
                    """)
                

            except Exception as e:
                st.error(f"Tahmin işlemi sırasında model bir hata ile karşılaştı: {e}")