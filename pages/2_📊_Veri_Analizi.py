import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Veri Analizi | Akıllı Stok", layout="wide")
st.title("🔍 Geçmiş Veri Analizi (EDA)")
st.markdown("Bu sayfa gerçek verileri derinlemesine analiz eder.")

sns.set_theme(style="whitegrid")

# 1. VERİYİ DOĞRUDAN OKU
df = pd.read_csv("data/temiz_veri.csv")

# 2. TEMEL İSTATİSTİKLER
st.subheader("📊 Veri Setine Genel Bakış")
col1, col2 = st.columns(2)

with col1:
    st.write("**İlk 5 Gözlem:**")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.write("**İstatistiksel Özet:**")
    st.dataframe(df.describe(), use_container_width=True)

st.divider()

st.subheader("📈 Kapsamlı Satış Analizleri")

# --- BİRİNCİ SATIR ---
col_kat, col_gun = st.columns(2)

with col_kat:
    st.write("**En Çok Satan 7 Kategori**")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # 'cat_' ile başlayan sütunları bul, topla, sırala ve ilk 7'sini al
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    populer_kat = df[cat_cols].sum().sort_values(ascending=False).head(7)
    populer_kat.index = populer_kat.index.str.replace('cat_', '') # 'cat_' yazısını sil
    
    sns.barplot(x=populer_kat.values, y=populer_kat.index, ax=ax1, palette='magma')
    ax1.set_xlabel("Toplam Satış Adedi")
    ax1.set_ylabel("Kategoriler")
    st.pyplot(fig1)

with col_gun:
    st.write("**Haftanın Günlerine Göre Sipariş Yoğunluğu**")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    
    # Günleri Türkçeye çevirip sıralı gösteriyoruz
    gunler = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    gun_isimleri = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    gun_dagilimi = df['purchase_date'].dt.day_name().value_counts().reindex(gunler).fillna(0)
    
    sns.barplot(x=gun_isimleri, y=gun_dagilimi.values, ax=ax2, palette='coolwarm')
    ax2.set_ylabel("Sipariş Adedi")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

st.divider()

# --- İKİNCİ SATIR ---
col_kde, col_corr = st.columns(2)

with col_kde:
    st.write("**Satış Yoğunluk Dağılımı**")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.kdeplot(df["total_price"], fill=True, ax=ax3, color='green')
    ax3.set_xlabel("Toplam Satış (TL)")
    st.pyplot(fig3)

with col_corr:
    st.write("**Korelasyon Matrisi**")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax4, annot=False)
    st.pyplot(fig4)

st.divider()

# --- ÜÇÜNCÜ SATIR ---
col_scatter1, col_scatter2 = st.columns(2)

with col_scatter1:
    st.write("**Birim Fiyat vs Toplam Satış İlişkisi**")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["unit_price"], y=df["total_price"], alpha=0.5, ax=ax5)
    ax5.set_xlabel("Birim Fiyat (TL)")
    ax5.set_ylabel("Toplam Satış (TL)")
    st.pyplot(fig5)

with col_scatter2:
    st.write("**Müşteri Yaşı vs Toplam Satış**")
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=df["customer_age"], y=df["total_price"], alpha=0.5, color='orange', ax=ax6)
    ax6.set_xlabel("Müşteri Yaşı")
    ax6.set_ylabel("Toplam Satış (TL)")
    st.pyplot(fig6)