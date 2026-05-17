import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# --- ARAYÜZ KONFİGÜRASYONU VE BAŞLIK ---
st.set_page_config(page_title="Akıllı Stok | Karar Destek Sistemi", layout="wide")

st.title("🔮 Akıllı Talep Tahmini ve Karar Destek Sistemi")
st.markdown("""
Bu panel, makine öğrenmesi algoritmalarını (XGBoost) kullanarak seçilen ürünün geçmiş trendlerini analiz eder. 
Dinamik zaman filtresi sayesinde hem **geçmişe dönük test (Backtesting)** yapabilir hem de **gelecek 7 günlük talebi** öngörebilirsiniz.
""")

# --- MODEL VE VERİ SETİ YÜKLEME ---
@st.cache_resource
def load_model_and_data():
    try:
        csv_path = os.path.join("data", "synthetic_ecommerce_dataset.csv") # Kendi dosya yoluna göre güncelle
        model_path = os.path.join("data", "xgboost_demand_forecasting.pkl")
        df = pd.read_csv(csv_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return df, model
    except Exception as e:
        return None, str(e)

# --- ÖZELLİK MÜHENDİSLİĞİ (PIPELINE) ---
@st.cache_data
def build_daily_sales(df):
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

# --- HİBRİT TAHMİN MOTORU (GÜNCELLENMİŞ) ---
def hybrid_forecast(model, daily_sales, product, secilen_tarih, n_days=7):
    secilen_tarih_pd = pd.to_datetime(secilen_tarih)
    en_son_tarih_pd = daily_sales['purchase_date'].max()
    
    product_data = daily_sales[daily_sales['product'] == product].copy()
    if product_data.empty: return None
        
    predictions = []
    gecmis_veri = product_data[product_data['purchase_date'] <= secilen_tarih_pd]
    if gecmis_veri.empty: return None
        
    recent_sales = list(gecmis_veri['quantity'].values[-30:])
    current_row = gecmis_veri.iloc[-1].copy()
    
    # 🚨 DÜZELTME 1: Trend kaymasını çözüyoruz
    urun_sayisi = len(daily_sales['product'].unique()) 
    
    for step in range(1, n_days + 1):
        target_date = secilen_tarih_pd + pd.Timedelta(days=step)
        
        # --- DURUM 1: GEÇMİŞ TEST (JUPYTER İLE BİREBİR AYNI) ---
        if target_date <= en_son_tarih_pd:
            gercek_satir = product_data[product_data['purchase_date'] == target_date]
            if not gercek_satir.empty:
                X = gercek_satir[FEATURE_COLS]
                pred_log = model.predict(X)[0]
                pred_real = float(np.clip(np.expm1(pred_log), 0, None))
                gercek_satis_adedi = int(gercek_satir.iloc[0]['quantity']) # Gerçek satış değerini çekiyoruz
                
                predictions.append({
                    'Tarih': target_date,
                    'Gün': target_date.strftime('%A'),
                    'Tahmin': round(pred_real),  # Tertemiz tam sayı
                    'Gerçekleşen Satış': int(gercek_satir.iloc[0]['quantity']),
                    'Durum': 'Geçmiş Test'
                })
                recent_sales.append(gercek_satir.iloc[0]['quantity'])
                current_row = gercek_satir.iloc[0].copy()
            else:
                break
                
        # --- DURUM 2: GELECEK TAHMİNİ (RECURSIVE) ---
        else:
            new_row = current_row.copy()
            new_row['purchase_date'] = target_date
            new_row['day'] = target_date.day
            new_row['month'] = target_date.month
            new_row['weekday'] = target_date.weekday()
            new_row['weekofyear'] = int(target_date.isocalendar().week)
            new_row['is_weekend'] = 1 if target_date.weekday() >= 5 else 0
            new_row['trend'] = current_row['trend'] + urun_sayisi # 🚨 Trendi ürün sayısı kadar artırıyoruz
            
            n = len(recent_sales)
            new_row['lag_1'] = recent_sales[-1] if n >= 1 else 0
            new_row['lag_2'] = recent_sales[-2] if n >= 2 else 0
            new_row['lag_3'] = recent_sales[-3] if n >= 3 else 0
            new_row['lag_7'] = recent_sales[-7] if n >= 7 else 0
            new_row['lag_14'] = recent_sales[-14] if n >= 14 else 0
            new_row['lag_21'] = recent_sales[-21] if n >= 21 else 0
            new_row['lag_30'] = recent_sales[-30] if n >= 30 else 0
            
            for w in [3, 7, 14, 30]:
                window_data = recent_sales[-w:] if n >= w else recent_sales
                new_row[f'rolling_mean_{w}'] = np.mean(window_data)
                new_row[f'rolling_std_{w}'] = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
            
            X = pd.DataFrame([new_row])[FEATURE_COLS]
            pred_log = model.predict(X)[0]
            pred_real = float(np.clip(np.expm1(pred_log), 0, None))
            
            predictions.append({
                'Tarih': target_date,
                'Gün': target_date.strftime('%A'),
                'Tahmin': round(pred_real), # Tertemiz tam sayı
                'Gerçekleşen Satış': None, 
                'Durum': 'Gelecek Projeksiyonu'
            })
            
            recent_sales.append(pred_real)
            current_row = new_row
            
    return pd.DataFrame(predictions)

# --- UYGULAMA MANTIĞI ---
res = load_model_and_data()
if isinstance(res, tuple) and res[0] is not None:
    df, model = res
else:
    st.error(f"🚨 Dosyalar yüklenirken hata oluştu: {res}")
    st.stop()

# --- ZAMAN FİLTRESİ VE TEST SETİ HESAPLAMALARI ---
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
en_erken_tarih = df['purchase_date'].min()
en_son_tarih = df['purchase_date'].max()

# DİKKAT: Ayşe'nin dropna() mantığıyla uyuşması için ilk 30 günü atlayarak günleri sayıyoruz!
gecerli_tarihler = pd.date_range(start=en_erken_tarih + pd.Timedelta(days=30), end=en_son_tarih, freq='D')

# Ayşe ile BİREBİR aynı test başlangıç gününü buluyoruz
split_index = int(len(gecerli_tarihler) * 0.80)
test_baslangic_tarihi = gecerli_tarihler[split_index].date()

# --- KULLANICI GİRDİ PANELİ (SIDEBAR) ---
st.sidebar.header("📅 Zaman Filtresi")

hedef_tarih = st.sidebar.date_input(
    "🎯 Tahmin Edilecek İlk Gün",
    value=en_son_tarih.date(),          
    min_value=test_baslangic_tarihi, # Kullanıcı test verisinin ilk gününden öncesini ASLA seçemez
    max_value=en_son_tarih.date() + pd.Timedelta(days=1)       
)
secilen_tarih = hedef_tarih - pd.Timedelta(days=1)

st.sidebar.caption(f"Geçmiş tahminler için en erken {test_baslangic_tarihi.strftime('%d.%m.%Y')} tarihi seçilebilir.")

st.sidebar.divider()

st.sidebar.header("📦 Ürün Filtresi")
kategoriler = sorted(df['category'].unique())
secilen_kat = st.sidebar.selectbox("🏷️ Kategori", kategoriler)
kategori_urunleri = df[df['category'] == secilen_kat]['product'].unique()
secilen_urun = st.sidebar.selectbox("📦 Tahmin Edilecek Ürün", sorted(kategori_urunleri))

gun_tr = {"Monday": "Pazartesi", "Tuesday": "Salı", "Wednesday": "Çarşamba",
          "Thursday": "Perşembe", "Friday": "Cuma", "Saturday": "Cumartesi", "Sunday": "Pazar"}

st.info(f"📍 **Seçilen Milat Tarihi:** {secilen_tarih.strftime('%d.%m.%Y')} — Model, bu tarihe kadar olan verileri baz alarak sonraki 7 günü hesaplayacaktır.")

if st.button("🚀 7 Günlük Analizi Başlat", use_container_width=True, type="primary"):
    with st.spinner("Model seçilen tarihe göre geçmişi derliyor ve gelecek 7 günü hesaplıyor..."):
        daily_sales = build_daily_sales(df)
        
        # Seçilen tarihi fonksiyona gönderiyoruz!
        tahmin_df = hybrid_forecast(model, daily_sales, secilen_urun, secilen_tarih, n_days=7)
        
        if tahmin_df is None or tahmin_df.empty:
            st.error("Seçilen ürün ve tarih aralığı için yeterli veri bulunamadı.")
        else:
            secilen_tarih_pd = pd.to_datetime(secilen_tarih)
            gecmis = daily_sales[(daily_sales['product'] == secilen_urun) & (daily_sales['purchase_date'] <= secilen_tarih_pd)]
            son_30_gun = gecmis.tail(30)
            
            ort_satis = son_30_gun['quantity'].mean() if not son_30_gun.empty else 1.0
            if pd.isna(ort_satis) or ort_satis == 0:
                ort_satis = 0.5
            
            toplam_7gun = tahmin_df['Tahmin'].sum()
            haftalik_ort = ort_satis * 7 
            
            if toplam_7gun >= haftalik_ort * 1.3:
                stok_karari, mesaj_tipi = "🟢 Stok Artır", "success"
            elif toplam_7gun >= haftalik_ort * 0.7:
                stok_karari, mesaj_tipi = "🟡 Stok Koru", "warning"
            else:
                stok_karari, mesaj_tipi = "🔴 Stok Azalt", "error"
            
            # --- ÜST METRİKLER ---
            st.divider()
            st.subheader(f"📊 {secilen_urun} — Tahmin Analizi")
            
            yarinki_tahmin = tahmin_df.iloc[0]['Tahmin'] 
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(label="🔜 İlk Gün Tahmini (T+1)", value=f"{yarinki_tahmin} Adet", help="Seçilen tarihten tam 1 gün sonrasının tahmini (Jupyter Notebook'taki test değerleriyle aynıdır).")
            c2.metric(label="📦 7 Günlük Toplam (T+7)", value=f"{toplam_7gun} Adet")
            c3.metric(label="⚙️ Stok Aksiyonu", value=stok_karari)
            c4.metric(label="📈 Geçmiş Haftalık Ort.", value=f"{haftalik_ort:.0f} Adet", delta=f"{toplam_7gun - haftalik_ort:+.0f} fark")
            
            # --- TABLO ---
            st.markdown("#### 📋 Günlük Tahmin Detayı")
            tablo = tahmin_df.copy() # Orijinal veriyi bozmuyoruz
            tablo['Tarih'] = tablo['Tarih'].dt.strftime('%d.%m.%Y')
            tablo['Gün'] = tablo['Gün'].map(gun_tr)
            
            # Süsleme işlemlerini sadece ekranda görünecek kopyaya uyguluyoruz
            tablo['Tahmin'] = tablo['Tahmin'].apply(lambda x: f"{x} Adet")
            tablo['Gerçekleşen Satış'] = tablo['Gerçekleşen Satış'].apply(
                lambda x: f"{int(x)} Adet" if pd.notna(x) else "Henüz Yaşanmadı"
            )
            
            tablo.index = range(1, len(tablo) + 1)
            tablo.index.name = "Gün #"
            st.dataframe(tablo, use_container_width=True)
            
            # --- GRAFİK ---
            st.markdown("### 📉 30 Günlük Geçmiş + 7 Günlük Projeksiyon")
            fig, ax = plt.subplots(figsize=(14, 5))
            
            ax.plot(son_30_gun['purchase_date'], son_30_gun['quantity'], marker='o', markersize=4, linestyle='-', linewidth=2, color='#2980b9', label='Geçmiş Satışlar (Gerçek)')
            ax.fill_between(son_30_gun['purchase_date'], son_30_gun['quantity'], alpha=0.1, color='#2980b9')
            
            tahmin_tarihleri = tahmin_df['Tarih']
            tahmin_degerleri = tahmin_df['Tahmin']
            
            if not son_30_gun.empty:
                son_gercek_tarih = son_30_gun.iloc[-1]['purchase_date']
                son_gercek_satis = son_30_gun.iloc[-1]['quantity']
                ax.plot([son_gercek_tarih, tahmin_tarihleri.iloc[0]], [son_gercek_satis, tahmin_degerleri.iloc[0]], linestyle='--', linewidth=1.5, color='#e74c3c', alpha=0.5)
            
            ax.plot(tahmin_tarihleri, tahmin_degerleri, marker='*', markersize=12, linestyle='-', linewidth=2, color='#e74c3c', label='7 Günlük Gelecek Tahmini')
            ax.fill_between(tahmin_tarihleri, tahmin_degerleri, alpha=0.1, color='#e74c3c')
            
            for tarih, deger in zip(tahmin_tarihleri, tahmin_degerleri):
                if deger > 0:
                    ax.annotate(f"{deger}", xy=(tarih, deger), xytext=(0, 8), textcoords='offset points', color='#e74c3c', fontweight='bold', fontsize=8, ha='center')
            
            ax.set_ylabel("Satış Adedi", fontsize=10)
            ax.tick_params(axis='x', rotation=35, labelsize=8)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
            
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            plt.tight_layout()
            st.pyplot(fig)