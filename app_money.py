# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="미국 주식 중기 포트폴리오", layout="wide")
st.title("📈 60/40 + 이동평균(선택) 리스크 관리 포트폴리오")

# ----- 사이드바: 파라미터 -----
with st.sidebar:
    st.header("설정")
    start = st.date_input("시작일", value=pd.to_datetime("2007-01-01"))
    equity = st.text_input("주식 ETF (대형주)", "SPY").upper().strip()
    bond = st.text_input("채권 ETF (종합채권)", "AGG").upper().strip()
    tbill = st.text_input("현금 대용(단기국채)", "BIL").upper().strip()

    eq_w = st.slider("주식 비중", 0.0, 1.0, 0.6, 0.05)
    bd_w = st.slider("채권 비중", 0.0, 1.0, 0.4, 0.05)
    st.caption("💡 두 비중 합이 1.0이 되도록 권장합니다.")

    rebalance = st.selectbox("리밸런싱 주기", ["M(월)", "Q(분기)", "A(연)"], index=1)
    freq_map = {"M(월)": "M", "Q(분기)": "Q", "A(연)": "A"}
    ma_window = st.number_input("이동평균(일)", min_value=20, max_value=400, value=200, step=5)
    use_filter = st.checkbox("리스크 필터 사용 (주식 < MA 이면 주식비중을 현금으로)", value=True)

    st.markdown("---")
    st.caption("데이터: Yahoo Finance (yfinance) / 교육용 예시")

@st.cache_data(show_spinner=False)
def load_prices(tickers, start):
    df = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    # 단일 티커 대응
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how="all")

def perf_stats(returns, freq=252):
    cagr = (1 + returns).prod() ** (freq / len(returns)) - 1
    vol = returns.std() * np.sqrt(freq)
    sharpe = (returns.mean() * freq) / vol if vol > 0 else np.nan
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    mdd = (cum / peak - 1).min()
    return cagr, vol, sharpe, mdd, cum

def backtest(px, equity, bond, tbill, eq_w, bd_w, rebalance, ma_window, use_filter):
    rets = px.pct_change().dropna()
    # 이동평균 시그널
    ma = px[equity].rolling(ma_window).mean()
    signal = (px[equity] >= ma).reindex(rets.index, method="ffill")

    # 리밸런싱 달력
    rdates = rets.resample(rebalance).last().index

    tickers = [equity, bond, tbill]
    w = pd.DataFrame(0.0, index=rets.index, columns=tickers)

    for dt in rdates:
        if use_filter:
            on = bool(signal.reindex([dt], method="ffill").iloc[0])
            if on:
                we, wb, wt = eq_w, bd_w, 0.0
            else:
                we, wb, wt = 0.0, bd_w, eq_w   # 주식 비중을 현금(tbill)로
        else:
            we, wb, wt = eq_w, bd_w, 0.0
        w.loc[dt, [equity, bond, tbill]] = [we, wb, wt]

    w = w.replace(0, np.nan).ffill().fillna(0)
    port_rets = (w * rets).sum(axis=1)

    # 비교: 순수 60/40
    w6040 = pd.DataFrame(0.0, index=rets.index, columns=tickers)
    w6040.loc[rdates, [equity, bond]] = [eq_w, bd_w]
    w6040 = w6040.replace(0, np.nan).ffill().fillna(0)
    rets6040 = (w6040 * rets).sum(axis=1)

    return port_rets, rets6040

# ----- 데이터 로드 -----
tickers = list({equity, bond, tbill})
px = load_prices(tickers, start)

# 유효성 체크
missing = [t for t in [equity, bond, tbill] if t not in px.columns]
if missing:
    st.error(f"가격 데이터가 없는 티커: {', '.join(missing)}")
    st.stop()

# ----- 백테스트 -----
port_rets, base_rets = backtest(
    px, equity, bond, tbill, eq_w, bd_w, freq_map[rebalance], ma_window, use_filter
)

# 성과지표
cagr, vol, sharpe, mdd, curve = perf_stats(port_rets)
cagr0, vol0, sharpe0, mdd0, curve0 = perf_stats(base_rets)

# ----- 출력 -----
colA, colB, colC, colD = st.columns(4)
colA.metric("CAGR", f"{cagr:.2%}", f"{cagr - cagr0:+.2%}")
colB.metric("Volatility", f"{vol:.2%}", f"{vol - vol0:+.2%}")
colC.metric("Sharpe", f"{sharpe:.2f}", f"{sharpe - sharpe0:+.2f}")
colD.metric("Max Drawdown", f"{mdd:.2%}", f"{mdd - mdd0:+.2%}")

st.markdown("### 자산곡선")
fig = plt.figure(figsize=(10,4))
plt.plot(curve.index, curve.values, label="전략: 60/40 + MA필터" if use_filter else "전략: 60/40")
plt.plot(curve0.index, curve0.values, label="비교: 60/40(필터 없음)", linestyle="--")
plt.legend(); plt.grid(True); plt.tight_layout()
st.pyplot(fig)

with st.expander("세부 로그"):
    st.write("리밸런싱:", rebalance, "/ 이동평균:", ma_window, "/ 필터 사용:", use_filter)
    st.dataframe(px.tail())
