import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Anasayfa | Akıllı Stok", layout="wide")
st.sidebar.markdown("### 📦 E-Ticaret Akıllı Stok")
st.title("E-Ticaret Akıllı Stok Yönetim Sistemi")

df = pd.read_csv("data/temiz_veri.csv")

# GERÇEK METRİKLERİ ONE-HOT SÜTUNLARDAN HESAPLA
toplam_satis = len(df)
ort_tutar = df['unit_price'].mean()

# Ürün çeşitliliğini 'prod_' ile başlayan sütunları sayarak bulur
prod_cols = [col for col in df.columns if col.startswith('prod_')]
urun_cesidi = len(prod_cols)

# En popüler kategoriyi 'cat_' ile başlayan sütunların toplamına bakarak bulur
cat_cols = [col for col in df.columns if col.startswith('cat_')]
en_populer_kat = df[cat_cols].sum().idxmax().replace('cat_', '')

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Satış", f"{toplam_satis:,} Adet")
col2.metric("Ort. Ürün Fiyatı", f"{ort_tutar:,.0f} TL")
col3.metric("Ürün Çeşidi", f"{urun_cesidi} Çeşit")
col4.metric("En Popüler Kategori", en_populer_kat)

st.divider()

# 3. GERÇEK GRAFİKLER
st.subheader("📈 Satış Performans Özetleri")
col_grafik1, col_grafik2 = st.columns(2)

with col_grafik1:
    st.write("**Aylık Satış Trendi**")
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    aylik_trend = df.groupby(df['purchase_date'].dt.to_period('M')).size()
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    aylik_trend.plot(kind='line', marker='o', ax=ax1, color='#2980b9', linewidth=2)
    ax1.set_xlabel("Tarih (Yıl-Ay)")
    ax1.set_ylabel("Sipariş Adedi")
    plt.tight_layout()
    st.pyplot(fig1)

with col_grafik2:
    st.write("**Ödeme Yöntemi Tercihleri**")
    odeme_dagilim = df.groupby('payment_method')['total_price'].sum().sort_values(ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.barplot(x=odeme_dagilim.index, y=odeme_dagilim.values, ax=ax2, palette='viridis')
    ax2.set_xlabel("Ödeme Yöntemi")
    ax2.set_ylabel("Toplam Gelir (TL)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)