import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Model Performansı | Akıllı Stok", layout="wide")
st.title("⚙️ XGBoost Modeli — Başarım Metrikleri")
st.markdown("E-ticaret talep tahmininde kullandığımız XGBoost zaman serisi modelinin akademik metrikleri ve karar mekanizması.")

# ─── METRİKLER ────────────────────────────────────────────────────────────────
st.header("📊 Test Seti Başarım Metrikleri")
st.caption("Modelin, eğitim verisini hiç görmediği %20'lik kronolojik test seti üzerindeki gerçek performansı:")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📈 R² Skoru", value="96.37%", delta="Log uzayında")
    st.caption("Tahmin edilen değerlerin gerçek değişkenliği ne kadar açıkladığı. **Önemli not:** Metrik log-transform sonrası hesaplanmıştır (bkz. aşağı).")
with col2:
    st.metric(label="🎯 MAE", value="0.12", delta="Log uzayında ✓", delta_color="inverse")
    st.caption("Log uzayındaki ortalama mutlak hata. Gerçek ölçekte yaklaşık **0.13 adet** günlük sapmaya karşılık gelir.")
with col3:
    st.metric(label="📉 RMSE", value="0.37", delta="Log uzayında ✓", delta_color="inverse")
    st.caption("Aykırı tahminlere karşı ağırlıklı hata. Gerçek ölçekte yaklaşık **0.45 adet** sapmaya eşdeğerdir.")

st.divider()

# ─── SKORLAR HAKKINDA DÜRÜST AÇIKLAMA ─────────────────────────────────────────
st.subheader("🔬 Metrikler Neden Bu Kadar İyi? (Dürüst Teknik Açıklama)")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Aralıklı Talep ve Log Transform**")
    st.write(
        "Veri setinde bir ürün **günlerin %71'inde hiç satılmıyor**, "
        "satıldığı gün ise ortalama **3.54 adet** satılıyor. "
        "Bu 'aralıklı talep' kalıbı e-ticarette yaygındır. "
        "Hedef değişken `log(quantity + 1)` dönüşümüyle eğitilmiştir — "
        "bu, sıfır-ağırlıklı dağılımın modeli bozmasını önleyen "
        "endüstri standardı bir tekniktir."
    )
with col_b:
    st.markdown("**Veri Sızıntısı (Leakage) Sıfır**")
    st.write(
        "Veri setinin **%80'i eğitim**, **%20'si test** olarak **kronolojik sırada** bölünmüştür. "
        "Model gelecekteki tarihlere ait hiçbir veriyi eğitim sürecinde görmemiştir. "
        "Ayrıca özellik mühendisliğinde kullanılan tüm Lag ve Rolling değerler "
        "yalnızca **geçmiş gözlemlere** bakarak hesaplanmıştır. "
        "Web arayüzünde 7 günlük tahmin, **özyinelemeli (recursive) yöntemle** üretilir."
    )

st.divider()

# ─── MİMARİ ───────────────────────────────────────────────────────────────────
st.subheader("🏗️ Model Mimarisi ve Özellik Mühendisliği")

col_c, col_d = st.columns([1, 2])
with col_c:
    st.markdown("**XGBoost Hiper-Parametreler**")
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42
    }
    st.dataframe(pd.DataFrame(params.items(), columns=["Parametre", "Değer"]),
                 use_container_width=True, hide_index=True)

with col_d:
    st.markdown("**22 Özellik Grubu**")
    ozellikler = pd.DataFrame({
        "Grup": ["Zamansal", "Zamansal", "Zamansal", "Zamansal", "Zamansal", "Zamansal",
                 "Lag (Geçmiş Satış)", "Lag (Geçmiş Satış)", "Lag (Geçmiş Satış)", "Lag (Geçmiş Satış)",
                 "Rolling Mean", "Rolling Mean", "Rolling Mean", "Rolling Mean",
                 "Rolling Std", "Rolling Std", "Rolling Std", "Rolling Std",
                 "Ürün", "Zamansal", "Zamansal", "Zamansal"],
        "Özellik": ["Gün", "Ay", "Haftanın Günü", "Hafta No", "Hafta Sonu?", "Zaman Trendi",
                    "Lag 1 (Dün)", "Lag 7 (Geçen Hafta)", "Lag 14", "Lag 30 (Geçen Ay)",
                    "Rolling Mean 3G", "Rolling Mean 7G", "Rolling Mean 14G", "Rolling Mean 30G",
                    "Rolling Std 3G", "Rolling Std 7G", "Rolling Std 14G", "Rolling Std 30G",
                    "Ürün (Label Encoded)", "Lag 2", "Lag 3", "Lag 21"]
    })
    st.dataframe(ozellikler, use_container_width=True, hide_index=True)

st.divider()

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
st.subheader("🧠 Modelin Karar Mekanizması (Feature Importance)")
st.write("Yapay zekanın tahminde bulunurken hangi özelliklere ne kadar ağırlık verdiği (En önemli 15 özellik):")

try:
    with open(os.path.join("data", "xgboost_demand_forecasting.pkl"), "rb") as f:
        model_xgboost = pickle.load(f)

    feature_names = [
        'Ürün (Encoded)', 'Gün', 'Ay', 'Haftanın Günü', 'Hafta No',
        'Hafta Sonu?', 'Zaman Trendi', 'Lag 1 (Dün)', 'Lag 2 (2G)',
        'Lag 3 (3G)', 'Lag 7 (1 Hafta)', 'Lag 14 (2 Hafta)',
        'Lag 21 (3 Hafta)', 'Lag 30 (1 Ay)',
        'Rolling Ort. 3G', 'Rolling Ort. 7G', 'Rolling Ort. 14G', 'Rolling Ort. 30G',
        'Rolling Std. 3G', 'Rolling Std. 7G', 'Rolling Std. 14G', 'Rolling Std. 30G'
    ]
    importances = model_xgboost.feature_importances_
    fi_df = pd.DataFrame({'Özellik': feature_names, 'Önem': importances})
    fi_df = fi_df.sort_values('Önem', ascending=True).tail(15)

    q75 = fi_df['Önem'].quantile(0.75)
    colors = ['#e74c3c' if v >= q75 else '#2980b9' for v in fi_df['Önem']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(fi_df['Özellik'], fi_df['Önem'], color=colors, edgecolor='none')
    ax.set_xlabel("Karara Etki Ağırlığı (Gain)")

    for bar, val in zip(bars, fi_df['Önem']):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    red_patch = mpatches.Patch(color='#e74c3c', label='En Kritik Özellikler (Top %25)')
    blue_patch = mpatches.Patch(color='#2980b9', label='Diğer Özellikler')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right')

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Model yüklenemedi: {e}")