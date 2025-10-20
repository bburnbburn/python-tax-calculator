# 필요한 라이브러리 설치 (Colab/로컬 최초 1회)
!pip install yfinance pandas numpy matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- 파라미터 --------
START = "2007-01-01"   # 금융위기 이전부터 테스트
END = None             # 오늘까지
EQUITY = "SPY"         # 미국 대형주 ETF
BOND = "AGG"           # 미국 종합채권 ETF
TBILL = "BIL"          # 미국 단기국채(현금 대용)
EQ_WEIGHT = 0.60       # 주식 비중
BD_WEIGHT = 0.40       # 채권 비중
REBAL_FREQ = "Q"       # 분기 리밸런싱(Q: 분기, M: 월, A: 연)
MA_WINDOW = 200        # 이동평균(리스크 필터)
USE_TREND_FILTER = True # True이면 SPY<200MA일 때 주식 비중을 BIL로 대체

# -------- 데이터 다운로드 --------
tickers = [EQUITY, BOND, TBILL]
px = yf.download(tickers, start=START, end=END, auto_adjust=True)["Close"].dropna()
rets = px.pct_change().dropna()

# 200일 이동평균과 필터 시그널
ma = px[EQUITY].rolling(MA_WINDOW).mean()
signal_equity = (px[EQUITY] >= ma).reindex(rets.index, method="ffill")  # True=주식 유지, False=현금(BIL)

# -------- 목표 비중(리밸런싱 달력) --------
# 리밸런싱 시점(분기 말)을 기준으로 타겟 가중치 테이블 생성
rebalance_dates = rets.resample(REBAL_FREQ).last().index

target_w = pd.DataFrame(index=rets.index, columns=tickers, data=0.0)
for dt in rebalance_dates:
    if not USE_TREND_FILTER:
        w_eq, w_bd, w_tb = EQ_WEIGHT, BD_WEIGHT, 0.0
    else:
        # 주식이 200MA 아래면 주식 비중을 TBILL로 대체
        in_risk_on = bool(signal_equity.reindex([dt], method="ffill").iloc[0])
        if in_risk_on:
            w_eq, w_bd, w_tb = EQ_WEIGHT, BD_WEIGHT, 0.0
        else:
            w_eq, w_bd, w_tb = 0.0, BD_WEIGHT, EQ_WEIGHT  # 주식 60% -> TBILL로 이동

    target_w.loc[dt, [EQUITY, BOND, TBILL]] = [w_eq, w_bd, w_tb]

# 타겟 비중을 매일 앞으로 채워 적용
target_w = target_w.replace(0, np.nan).ffill().fillna(0)

# -------- 포트폴리오 수익률(가중합) --------
port_rets = (target_w * rets).sum(axis=1)

# 자산곡선
init_value = 1.0
equity_curve = (1 + port_rets).cumprod() * init_value

# -------- 성과지표 --------
def perf_stats(returns, freq=252):
    # 일간 수익률 기준
    cagr = (1 + returns).prod() ** (freq / len(returns)) - 1
    vol = returns.std() * np.sqrt(freq)
    sharpe = (returns.mean() * freq) / vol if vol > 0 else np.nan
    # 최대낙폭
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    mdd = dd.min()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd
    }

stats = perf_stats(port_rets)

# -------- 비교: 순수 60/40 (필터 없음) --------
w6040 = pd.DataFrame(index=rets.index, columns=tickers, data=0.0)
w6040.loc[rebalance_dates, [EQUITY, BOND]] = [EQ_WEIGHT, BD_WEIGHT]
w6040 = w6040.replace(0, np.nan).ffill().fillna(0)
rets_6040 = (w6040 * rets).sum(axis=1)
curve_6040 = (1 + rets_6040).cumprod()

stats_6040 = perf_stats(rets_6040)

# -------- 결과 출력 --------
print("=== 리스크 필터 적용 포트폴리오 ===")
for k, v in stats.items():
    print(f"{k:>12}: {v: .2%}")

print("\n=== 순수 60/40 비교(필터 없음) ===")
for k, v in stats_6040.items():
    print(f"{k:>12}: {v: .2%}")

# 그래프
plt.figure(figsize=(10,5))
plt.plot(equity_curve, label=f"60/40 + 200MA 필터({EQUITY}->{TBILL})" if USE_TREND_FILTER else "60/40")
plt.plot(curve_6040, label="60/40 (필터 없음)", linestyle="--")
plt.title("포트폴리오 자산곡선 (기준가=1)")
plt.legend()
plt.grid(True)
plt.show()
