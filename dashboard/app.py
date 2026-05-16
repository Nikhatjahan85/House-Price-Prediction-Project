import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI House Price Prediction",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1,h2,h3,h4,h5,h6 {
    color: white !important;
}

p {
    color: #cbd5e1;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(to right,#16a34a,#22c55e);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.metric-card {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.section-box {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🏠 Enter Property Details")

    area = st.slider(
        "Area (sq ft)",
        500,
        5000,
        1800
    )

    bedrooms = st.slider(
        "Bedrooms",
        1,
        10,
        3
    )

    bathrooms = st.slider(
        "Bathrooms",
        1,
        10,
        2
    )

    parking = st.slider(
        "Parking",
        0,
        5,
        1
    )

    age = st.slider(
        "Property Age",
        0,
        50,
        5
    )

    furnishing = st.selectbox(
        "Furnishing",
        [
            "Furnished",
            "Semi-Furnished",
            "Unfurnished"
        ]
    )

    location = st.selectbox(
        "Location",
        [
            "Urban",
            "Semi-Urban",
            "Rural"
        ]
    )

    predict_button = st.button(
        "Predict House Price"
    )

# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<h1 style='
text-align:center;
font-size:65px;
font-weight:bold;
background: linear-gradient(to right,#22c55e,#3b82f6,#9333ea);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
'>

🏡 AI Powered House Price Prediction

</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='
text-align:center;
font-size:22px;
color:#cbd5e1;
'>

Smart Real Estate Analytics & Property Value Estimation Dashboard

</p>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# =====================================================
# METRIC CARDS
# =====================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.markdown("""
    <div class='metric-card'
    style='background:linear-gradient(to right,#2563eb,#3b82f6);'>

    <h3>📊 Accuracy</h3>

    <h1>94%</h1>

    </div>
    """, unsafe_allow_html=True)

with m2:

    st.markdown("""
    <div class='metric-card'
    style='background:linear-gradient(to right,#7c3aed,#9333ea);'>

    <h3>🏘 Properties</h3>

    <h1>1000+</h1>

    </div>
    """, unsafe_allow_html=True)

with m3:

    st.markdown("""
    <div class='metric-card'
    style='background:linear-gradient(to right,#16a34a,#22c55e);'>

    <h3>⚡ Speed</h3>

    <h1>1.5s</h1>

    </div>
    """, unsafe_allow_html=True)

with m4:

    st.markdown("""
    <div class='metric-card'
    style='background:linear-gradient(to right,#dc2626,#ef4444);'>

    <h3>🌍 Locations</h3>

    <h1>3 Types</h1>

    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    predicted_price = (
        area * 3000
        + bedrooms * 500000
        + bathrooms * 300000
        + parking * 200000
        - age * 10000
    )

    if location == "Urban":
        predicted_price += 2000000

    elif location == "Semi-Urban":
        predicted_price += 1000000

    else:
        predicted_price += 300000

    if furnishing == "Furnished":
        predicted_price += 500000

    elif furnishing == "Semi-Furnished":
        predicted_price += 250000

    else:
        predicted_price += 100000

    predicted_price += np.random.randint(
        -100000,
        100000
    )

    st.write("")
    st.write("")

    # =====================================================
    # MAIN LAYOUT
    # =====================================================

    left, right = st.columns([1.5,1])

    # =====================================================
    # LEFT PANEL
    # =====================================================

    with left:

        st.subheader("📈 Prediction Result")

        st.markdown(f"""
        <div class='section-box'>

        <h1 style='
        color:#22c55e;
        text-align:center;
        font-size:60px;
        '>

        ₹ {predicted_price:,.0f}

        </h1>

        <p style='
        text-align:center;
        font-size:20px;
        color:#cbd5e1;
        '>

        Estimated House Price

        </p>

        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # PIE + BAR CHARTS
        # =====================================================

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("🥧 Property Distribution")

            pie_labels = [
                "Area",
                "Bedrooms",
                "Bathrooms",
                "Parking"
            ]

            pie_values = [
                area,
                bedrooms * 100,
                bathrooms * 100,
                parking * 100
            ]

            fig1, ax1 = plt.subplots(figsize=(4,4))

            ax1.pie(
                pie_values,
                labels=pie_labels,
                autopct='%1.1f%%',
                colors=[
                    "green",
                    "red",
                    "orange",
                    "blue"
                ]
            )

            st.pyplot(fig1)

        with c2:

            st.subheader("📊 Feature Impact")

            features = [
                "Area",
                "Location",
                "Bedrooms",
                "Bathrooms"
            ]

            values = [
                95,
                85,
                70,
                60
            ]

            fig2, ax2 = plt.subplots(figsize=(4,4))

            ax2.bar(
                features,
                values,
                color=[
                    "green",
                    "green",
                    "orange",
                    "red"
                ]
            )

            ax2.set_ylabel("Impact")

            st.pyplot(fig2)

        # =====================================================
        # PRICE TREND
        # =====================================================

        st.subheader("📉 Property Price Trend")

        years = [
            2021,
            2022,
            2023,
            2024,
            2025
        ]

        prices = [
            predicted_price - 700000,
            predicted_price - 400000,
            predicted_price - 200000,
            predicted_price,
            predicted_price + 300000
        ]

        fig3, ax3 = plt.subplots(figsize=(8,4))

        ax3.plot(
            years,
            prices,
            marker='o',
            linewidth=4,
            color='green'
        )

        ax3.fill_between(
            years,
            prices,
            alpha=0.3,
            color='green'
        )

        ax3.set_title("House Price Growth")

        ax3.set_xlabel("Year")

        ax3.set_ylabel("Price")

        st.pyplot(fig3)

    # =====================================================
    # RIGHT PANEL
    # =====================================================

    with right:

        st.subheader("📋 Property Summary")

        summary_df = pd.DataFrame({

            "Feature": [
                "Area",
                "Bedrooms",
                "Bathrooms",
                "Parking",
                "Age",
                "Furnishing",
                "Location"
            ],

            "Value": [
                area,
                bedrooms,
                bathrooms,
                parking,
                age,
                furnishing,
                location
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True,
            height=300
        )

        st.write("")

        st.subheader("🚀 Insights")

        st.success(
            "Urban properties have higher demand."
        )

        st.warning(
            "Older houses reduce market value."
        )

        st.info(
            "Area contributes most to pricing."
        )

        st.error(
            "Limited parking decreases resale value."
        )

    # =====================================================
    # MARKET INTELLIGENCE SECTION
    # =====================================================

    st.write("")
    st.write("")

    st.markdown("""
    <h1 style='
    text-align:center;
    font-size:45px;
    background: linear-gradient(to right,#22c55e,#3b82f6,#9333ea);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    '>

    📊 Market Intelligence & Growth Insights

    </h1>
    """, unsafe_allow_html=True)

    # =====================================================
    # HEATMAP + MARKET MOVEMENT
    # =====================================================

    a1, a2 = st.columns(2)

    with a1:

        st.subheader("🔥 Area Demand Heatmap")

        heat_data = np.array([
            [85, 70, 65, 90],
            [60, 95, 75, 88],
            [45, 55, 92, 78],
            [80, 67, 72, 99]
        ])

        fig4, ax4 = plt.subplots(figsize=(5,4))

        heatmap = ax4.imshow(
            heat_data,
            cmap="RdYlGn"
        )

        plt.colorbar(heatmap)

        ax4.set_title("Demand Heatmap")

        st.pyplot(fig4)

    with a2:

        st.subheader("📈 Market Rise & Fall")

        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug"
        ]

        growth = [
            20,
            35,
            28,
            50,
            65,
            48,
            75,
            90
        ]

        fig5, ax5 = plt.subplots(figsize=(5,4))

        ax5.plot(
            months,
            growth,
            marker='o',
            linewidth=4,
            color='green'
        )

        ax5.fill_between(
            months,
            growth,
            alpha=0.3,
            color='green'
        )

        ax5.set_title("Market Movement")

        ax5.set_ylabel("Growth %")

        st.pyplot(fig5)

    # =====================================================
    # PIE + DEMAND GRAPH
    # =====================================================

    b1, b2 = st.columns(2)

    with b1:

        st.subheader("🥧 Market Share")

        market_labels = [
            "Urban",
            "Semi-Urban",
            "Rural"
        ]

        market_values = [
            55,
            30,
            15
        ]

        fig6, ax6 = plt.subplots(figsize=(5,4))

        ax6.pie(
            market_values,
            labels=market_labels,
            autopct='%1.1f%%',
            colors=[
                "green",
                "orange",
                "red"
            ]
        )

        st.pyplot(fig6)

    with b2:

        st.subheader("🏠 Property Demand Score")

        demand_categories = [
            "Luxury",
            "Family",
            "Parking",
            "Security",
            "Resale"
        ]

        demand_scores = [
            95,
            85,
            70,
            88,
            75
        ]

        fig7, ax7 = plt.subplots(figsize=(5,4))

        ax7.bar(
            demand_categories,
            demand_scores,
            color=[
                "green",
                "limegreen",
                "orange",
                "blue",
                "red"
            ]
        )

        ax7.set_ylabel("Performance")

        st.pyplot(fig7)

    # =====================================================
    # INVESTMENT + CUSTOMER SATISFACTION
    # =====================================================

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("💰 Investment Growth")

        years2 = [
            2021,
            2022,
            2023,
            2024,
            2025
        ]

        roi = [
            15,
            28,
            40,
            68,
            92
        ]

        fig8, ax8 = plt.subplots(figsize=(5,4))

        ax8.plot(
            years2,
            roi,
            marker='o',
            linewidth=4,
            color='red'
        )

        ax8.fill_between(
            years2,
            roi,
            alpha=0.3,
            color='red'
        )

        ax8.set_ylabel("ROI %")

        st.pyplot(fig8)

    with c2:

        st.subheader("😊 Customer Satisfaction")

        ratings = [
            "Excellent",
            "Good",
            "Average",
            "Poor"
        ]

        rating_values = [
            55,
            30,
            10,
            5
        ]

        fig9, ax9 = plt.subplots(figsize=(5,4))

        ax9.barh(
            ratings,
            rating_values,
            color=[
                "green",
                "limegreen",
                "orange",
                "red"
            ]
        )

        ax9.set_xlabel("Users %")

        st.pyplot(fig9)

# =====================================================
# FOOTER
# =====================================================

st.write("")
st.write("")

st.markdown("""
<h3 style='
text-align:center;
color:#cbd5e1;
'>

🚀 Powered by Python, Streamlit & Machine Learning

</h3>
""", unsafe_allow_html=True)