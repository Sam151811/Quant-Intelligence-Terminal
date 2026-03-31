import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from utilities import (
    calculate_returns_and_cov, 
    portfolio_performance, 
    optimize_portfolio,
    calculate_efficient_frontier,
    calculate_advanced_metrics
)

st.set_page_config(page_title="Quant-Intelligence Terminal", layout="wide", page_icon="⚖️")

st.markdown("""
    <style>
    .stApp { background-color: #05070a; color: #ffffff; font-family: 'Inter', sans-serif; }
    
    div[data-testid="stTable"] { background-color: #111418 !important; border-radius: 8px; }
    div[data-testid="stTable"] td, div[data-testid="stTable"] th {
        color: #ffffff !important; font-weight: 600 !important;
        background-color: #111418 !important; border: 1px solid #333d47 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #111418; padding: 20px; border-radius: 10px; border-left: 6px solid #d4af37;
    }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div[data-testid="stMetricLabel"] { color: #d4af37 !important; font-weight: 700 !important; }

    .suggestion-box {
        background-color: #111418; padding: 20px; border: 1px solid #d4af37;
        border-radius: 8px; margin-bottom: 15px; color: #ffffff; font-size: 16px;
    }

    .main-header { font-size: 40px; font-weight: 900; color: #ffffff; margin: 0; }
    .sub-header { font-size: 14px; color: #d4af37; text-transform: uppercase; letter-spacing: 2px; }
    </style>
    """, unsafe_allow_html=True)

def simulate_opportunity_set(returns, cov, rf_rate, num_portfolios=2500):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(returns))
        weights /= np.sum(weights)
        p_ret, p_std, p_sharpe = portfolio_performance(weights, returns, cov, rf_rate)
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = p_sharpe
    return results

def run_monte_carlo(weights, returns_mean, cov_matrix, days=252, simulations=100):
    port_return = np.sum(returns_mean * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    random_rets = np.random.normal(port_return/252, port_vol/np.sqrt(252), (days, simulations))
    price_paths = 100 * (1 + random_rets).cumprod(axis=0)
    expected_path = 100 * (1 + port_return/252)**np.arange(days)
    return price_paths, expected_path

def main():
    st.markdown('<p class="main-header">Asset Intelligence Terminal</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Institutional Quantitative Engine</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Terminal Config")
        tickers = st.text_area("Asset Basket", value="AAPL, MSFT, NVDA, RELIANCE.NS, BTC-USD, GC=F")
        c1, c2 = st.columns(2)
        with c1: start_date = st.date_input("Start", datetime.date(2022, 1, 1))
        with c2: end_date = st.date_input("End", datetime.date.today())
        rf_rate = st.number_input("RF Rate (%)", 0.0, 10.0, 4.3) / 100
        run_btn = st.button("RUN ENGINE", use_container_width=True, type="primary")

    if run_btn or st.session_state.get('optimized', False):
        st.session_state['optimized'] = True
        t_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

        try:
            raw = yf.download(t_list, start=start_date, end=end_date)
            data = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
            data = data.dropna()
        except:
            st.error("Data Fetch Error"); st.stop()

        returns_annual, cov_matrix = calculate_returns_and_cov(data)
        ms_res = optimize_portfolio(returns_annual, cov_matrix, rf_rate, target='Sharpe')
        mv_res = optimize_portfolio(returns_annual, cov_matrix, rf_rate, target='Volatility')

        vol = np.sqrt(np.diag(cov_matrix))
        sharpe = returns_annual / vol
        
        # --- CONSULTANT FEATURE ---
        st.subheader("🧠 Intelligence Consultant")
        st.markdown(f"""
            <div class="suggestion-box">
                <b>🚀 Efficiency Leader:</b> {sharpe.idxmax()} is the top risk-adjusted performer.<br>
                <b>📉 Efficiency Laggard:</b> {sharpe.idxmin()} adds the most relative drag to the portfolio.<br>
                <b>⚖️ Systemic Correlation:</b> {data.pct_change().corr().values.mean():.2f} (Target < 0.40 for high diversification).
            </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 PERFORMANCE", "🎯 ALLOCATION", "🛡️ RISK FRONTIER", "🎲 PROJECTIONS", "🧮 DATA LAB"])

        with tab1:
            st.line_chart((data / data.iloc[0]) * 100, height=400)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                if ms_res.success:
                    r, v, s = portfolio_performance(ms_res.x, returns_annual, cov_matrix, rf_rate)
                    st.metric("Max Sharpe Return", f"{r:.2%}", delta=f"Sharpe: {s:.2f}")
                    st.plotly_chart(px.pie(names=data.columns, values=ms_res.x, hole=0.5, template="plotly_dark"), use_container_width=True)

        with tab3:
            st.subheader("Markowitz Opportunity Set & Efficient Frontier")
            
            # 1. Generate Cloud Points
            sim_data = simulate_opportunity_set(returns_annual, cov_matrix, rf_rate)
            
            # 2. Generate the mathematical Frontier Curve
            ef_results = calculate_efficient_frontier(returns_annual, cov_matrix, rf_rate)
            
            fig_ef = go.Figure()
            
            # The Cloud (Points)
            fig_ef.add_trace(go.Scatter(x=sim_data[0,:], y=sim_data[1,:], mode='markers', name='Simulations',
                                     marker=dict(color=sim_data[2,:], colorscale='Viridis', size=4, opacity=0.3)))
            
            # The Frontier Curve (Line)
            if ef_results:
                ef_vols = [portfolio_performance(p.x, returns_annual, cov_matrix, rf_rate)[1] for p in ef_results]
                ef_rets = [portfolio_performance(p.x, returns_annual, cov_matrix, rf_rate)[0] for p in ef_results]
                fig_ef.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode='lines', name='Efficient Frontier',
                                         line=dict(color='#d4af37', width=4)))

            # Individual Assets
            fig_ef.add_trace(go.Scatter(x=np.sqrt(np.diag(cov_matrix)), y=returns_annual, mode='markers+text', 
                                     text=data.columns, name='Assets', marker=dict(color='white', size=10)))
            
            # Optimal Marker
            if ms_res.success:
                r_ms, v_ms, _ = portfolio_performance(ms_res.x, returns_annual, cov_matrix, rf_rate)
                fig_ef.add_trace(go.Scatter(x=[v_ms], y=[r_ms], marker=dict(color='#00ffcc', size=15, symbol='star'), name='OPTIMAL'))

            fig_ef.update_layout(template="plotly_dark", xaxis_title="Risk (σ)", yaxis_title="Return (E)", height=600)
            st.plotly_chart(fig_ef, use_container_width=True)

        with tab4:
            st.subheader("Monte Carlo Path Simulation")
            paths, mean_line = run_monte_carlo(ms_res.x, returns_annual, cov_matrix)
            fig_mc = go.Figure()
            for i in range(50):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode='lines', line=dict(width=1.5, color='#444d56'), opacity=0.3, showlegend=False))
            fig_mc.add_trace(go.Scatter(y=mean_line, mode='lines', name='Expected Growth', line=dict(color='#d4af37', width=5)))
            fig_mc.update_layout(template="plotly_dark", xaxis_title="Days", yaxis_title="Portfolio Value")
            st.plotly_chart(fig_mc, use_container_width=True)

        with tab5:
            st.plotly_chart(px.imshow(data.pct_change().corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)

        st.divider()
        st.subheader("Strategic Benchmarking")
        daily_ret = data.pct_change().dropna()
        metrics = pd.DataFrame({
            "Max Sharpe": calculate_advanced_metrics((daily_ret * ms_res.x).sum(axis=1), rf_rate),
            "Min Volatility": calculate_advanced_metrics((daily_ret * mv_res.x).sum(axis=1), rf_rate)
        }).T
        st.table(metrics.style.format("{:.2f}"))

if __name__ == "__main__":
    main()