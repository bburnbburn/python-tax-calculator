
import io
import json
import math
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="부동산 이상거래 탐지", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
def to_py(m2: float) -> float:
    return m2 * 0.3025  # 1㎡ ≈ 0.3025평

def price_per_py(price: float, m2: float) -> float:
    py = to_py(m2) or 1.0
    return price / py

def stddev(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=1))

def key_of(row: pd.Series) -> str:
    return f"{row['complex']}__{row['area']}__{row['date']}"  # 월·단지·면적

@st.cache_data(show_spinner=False)
def read_json_records(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    # basic type fix
    num_cols = ["area","price","floor","buildYear"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _find_header_line(lines) -> Optional[int]:
    # Heuristics: look for a line that contains common headers including "시군구" and "단지명"
    for idx, line in enumerate(lines[:200]):
        if ("시군구" in line) and ("단지명" in line) and ("거래금액" in line):
            return idx
    # Fallback: known MOLIT export has header around line 15 (0-indexed)
    return 15

@st.cache_data(show_spinner=False)
def read_molit_csv(file_bytes: bytes) -> pd.DataFrame:
    # Try cp949 first (common for MOLIT export). Ignore errors.
    text = file_bytes.decode("cp949", errors="ignore")
    lines = text.splitlines()
    header_idx = _find_header_line(lines)
    # Rebuild clean CSV starting from header
    cleaned = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(cleaned),
        engine="python",
        quotechar='"',
        thousands=",",
    )
    need_cols = ["시군구","단지명","전용면적(㎡)","계약년월","계약일","거래금액(만원)","층","건축년도"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"CSV 헤더에 다음 컬럼이 없습니다: {miss}")
    # Parse region
    def parse_region(s: str) -> Tuple[str,str]:
        parts = str(s).split()
        if len(parts) >= 3:
            return parts[1], parts[2]  # 구, 동
        if len(parts) == 2:
            return parts[1], ""
        return parts[0] if parts else "", ""

    district, dong = zip(*df["시군구"].map(parse_region))
    out = pd.DataFrame({
        "district": district,
        "dong": dong,
        "complex": df["단지명"].astype(str),
        "area": pd.to_numeric(df["전용면적(㎡)"], errors="coerce"),
        "price": pd.to_numeric(df["거래금액(만원)"], errors="coerce"),
        "date": df["계약년월"].apply(lambda v: f"{int(v)//100:04d}-{int(v)%100:02d}" if pd.notna(v) else ""),
        "floor": pd.to_numeric(df["층"], errors="coerce"),
        "buildYear": pd.to_numeric(df["건축년도"], errors="coerce"),
    })
    out = out.dropna(subset=["area","price","floor","buildYear"]).reset_index(drop=True)
    out.insert(0, "id", np.arange(1, len(out)+1))
    return out

# -----------------------------
# Data loading
# -----------------------------
st.title("🏙️ 부동산 이상거래 탐지 시스템 (Streamlit)")
st.caption("동일 월·단지·면적 기준 통계 + Z-score 기반 정량 탐지")

col_left, col_right = st.columns([2,1])
with col_right:
    st.markdown("#### 데이터 소스")
    default_json_path = Path("daejeon_trades_2025.json")
    use_uploaded = st.toggle("업로드한 파일 사용(권장)", value=False, help="MOLIT CSV 또는 위 JSON 스키마 파일을 업로드해서 분석합니다.")
    uploaded = st.file_uploader("CSV(JSON) 업로드", type=["csv","json"], accept_multiple_files=False)

# Load logic
df: Optional[pd.DataFrame] = None
load_err: Optional[str] = None
if use_uploaded and uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".json"):
            df = pd.DataFrame(json.loads(uploaded.getvalue().decode("utf-8")))
            num_cols = ["area","price","floor","buildYear"]
            for c in num_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "id" not in df.columns:
                df.insert(0, "id", np.arange(1, len(df)+1))
        else:
            df = read_molit_csv(uploaded.getvalue())
    except Exception as e:
        load_err = f"업로드 파일 파싱 실패: {e}"
else:
    try:
        if default_json_path.exists():
            df = read_json_records(default_json_path)
            if "id" not in df.columns:
                df.insert(0, "id", np.arange(1, len(df)+1))
        else:
            load_err = "프로젝트 루트에 'daejeon_trades_2025.json' 파일을 두거나, CSV/JSON을 업로드하세요."
    except Exception as e:
        load_err = f"기본 JSON 읽기 실패: {e}"

if load_err:
    st.error(load_err)
    st.stop()

if df is None or df.empty:
    st.warning("데이터가 비어있습니다.")
    st.stop()

# Ensure column types
df["district"] = df["district"].astype(str)
df["dong"] = df["dong"].astype(str)
df["complex"] = df["complex"].astype(str)
df["date"] = df["date"].astype(str)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ 필터 & 설정")
months = ["전체"] + sorted([m for m in df["date"].dropna().unique().tolist() if m])
districts = ["전체"] + sorted(df["district"].dropna().unique().tolist())

selected_district = st.sidebar.selectbox("자치구", districts, index=0)
selected_month = st.sidebar.selectbox("월 선택", months, index=0)
min_score = st.sidebar.slider("민감도 (이상 신호 점수 ≥)", min_value=10, max_value=70, value=25, step=5)
use_z = st.sidebar.checkbox("Z-score 규칙 사용 (±2σ)", value=True)
sort_opt = st.sidebar.selectbox("정렬", ["위험도 높은 순","가격 높은 순","가격 낮은 순"], index=0)

# -----------------------------
# Group statistics (월·단지·면적)
# -----------------------------
mask = (selected_district == "전체") | (df["district"] == selected_district)
mask_month = (selected_month == "전체") | (df["date"] == selected_month)
dfv = df[mask & mask_month].copy()

# Build group stats
group_key = dfv.apply(lambda r: key_of(r), axis=1)
dfv["_gk"] = group_key
gp = dfv.groupby("_gk")["price"].agg(["mean","std","count"]).rename(columns={"mean":"avg","std":"std","count":"n"})
dfv = dfv.join(gp, on="_gk")

# Compute anomaly metrics
dfv["deviationPct"] = (dfv["price"] - dfv["avg"]) / dfv["avg"].replace(0, np.nan) * 100.0
dfv["py"] = (dfv["price"] / (dfv["area"] * 0.3025)).round()  # price per py
dfv["avgPy"] = (dfv["avg"] / (dfv["area"] * 0.3025)).round()
dfv["pyDeviationAbs"] = (dfv["py"] - dfv["avgPy"]).abs()
dfv["z"] = (dfv["price"] - dfv["avg"]) / dfv["std"].replace(0, np.nan)
dfv["z"] = dfv["z"].fillna(0.0)

# Scoring
score = np.zeros(len(dfv), dtype=float)
reasons = [[] for _ in range(len(dfv))]

up_mask = dfv["deviationPct"] > 30
down_mask = dfv["deviationPct"] < -20
low_floor_up = (dfv["floor"] <= 5) & (dfv["deviationPct"] > 10)
py_anom = dfv["pyDeviationAbs"] > (dfv["avgPy"] * 0.25)
z_anom = use_z & (dfv["z"].abs() >= 2)

score = score + np.where(up_mask, 40, 0)
score = score + np.where(down_mask, 35, 0)
score = score + np.where(low_floor_up, 15, 0)
score = score + np.where(py_anom, 20, 0)
score = score + np.where(z_anom, 25, 0)

def push_reason(i, cond, text):
    if cond.iloc[i]:
        reasons[i].append(text)

for i in range(len(dfv)):
    push_reason(i, up_mask, "급격한 가격 상승")
    push_reason(i, down_mask, "급격한 가격 하락")
    push_reason(i, low_floor_up, "저층 가격 이상")
    push_reason(i, py_anom, "평당가 이상")
    if use_z and abs(float(dfv.iloc[i]['z'])) >= 2:
        reasons[i].append(f"Z-score {dfv.iloc[i]['z']:.1f}σ 이상")

dfv["anomalyScore"] = np.minimum(100, np.round(score)).astype(int)
dfv["anomalyReasons"] = reasons
dfv["isAnomaly"] = dfv["anomalyScore"] >= min_score

# Sorting
if sort_opt == "가격 높은 순":
    dfv = dfv.sort_values(by=["price","anomalyScore"], ascending=[False, False])
elif sort_opt == "가격 낮은 순":
    dfv = dfv.sort_values(by=["price","anomalyScore"], ascending=[True, False])
else:
    dfv = dfv.sort_values(by=["isAnomaly","anomalyScore","price"], ascending=[False, False, False])

# Stats
anoms = dfv[dfv["isAnomaly"]]
high = int((anoms["deviationPct"] > 0).sum())
low = int((anoms["deviationPct"] < 0).sum())
rate = (len(anoms) / len(dfv) * 100.0) if len(dfv) else 0.0

# -----------------------------
# Top summary cards
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("분석 대상", f"{len(dfv):,} 건")
c2.metric("이상거래", f"{len(anoms):,} 건", f"{rate:.1f}%")
c3.metric("급등 의심", f"{high:,} 건")
c4.metric("급락 의심", f"{low:,} 건")

st.divider()

# -----------------------------
# Results Table
# -----------------------------
st.subheader(f"이상거래 탐지 결과 (필터 적용 후 {len(anoms):,}건)")
show_only_anom = st.toggle("이상거래만 보기", value=True)

columns_order = [
    "anomalyScore","district","complex","dong","area","price","py",
    "avg","avgPy","deviationPct","z","date","floor","buildYear","anomalyReasons"
]
df_show = dfv.copy()
if show_only_anom:
    df_show = df_show[df_show["isAnomaly"]]

def fmt_reason(xs):
    return " / ".join(xs) if isinstance(xs, list) else ""

out = df_show[columns_order].copy()
out = out.rename(columns={
    "anomalyScore": "위험도",
    "district": "자치구",
    "complex": "단지명",
    "dong": "동",
    "area": "면적(㎡)",
    "price": "거래가(만원)",
    "py": "평당가(만원)",
    "avg": "평균가(만원)",
    "avgPy": "평균 평당가(만원)",
    "deviationPct": "편차(%)",
    "z": "Z",
    "date": "월",
    "floor": "층",
    "buildYear": "건축년도",
    "anomalyReasons": "이상사유",
})
out["편차(%)"] = out["편차(%)"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "")
out["평당가(만원)"] = out["평당가(만원)"].map(lambda v: f"{int(v):,}")
out["평균 평당가(만원)"] = out["평균 평당가(만원)"].map(lambda v: f"{int(v):,}")
out["거래가(만원)"] = out["거래가(만원)"].map(lambda v: f"{int(v):,}")
out["평균가(만원)"] = out["평균가(만원)"].map(lambda v: f"{int(v):,}")
out["이상사유"] = df_show["anomalyReasons"].map(fmt_reason)

st.dataframe(out, use_container_width=True, height=600)

# -----------------------------
# Download
# -----------------------------
st.download_button(
    "현재 결과 다운로드 (CSV)",
    data=df_show.to_csv(index=False).encode("utf-8-sig"),
    file_name="anomaly_results.csv",
    mime="text/csv",
)
st.caption("※ 결과 CSV는 필터/민감도/Z-score 설정이 반영된 현재 화면 기준입니다.")

# -----------------------------
# Help
# -----------------------------
with st.expander("📘 탐지 알고리즘 설명", expanded=False):
    st.markdown(
        """
- **집계 기준**: 동일 **월·단지·면적** 그룹 평균 및 표준편차 사용
- **급격한 가격 상승**: 그룹 평균 대비 **+30%** 이상
- **급격한 가격 하락**: 그룹 평균 대비 **−20%** 이하
- **저층 가격 이상**: **5층 이하**이면서 **+10%** 초과
- **평당가 이상**: 그룹 평균 평당가 대비 **25%** 초과
- **Z-score**: \\(|Z| ≥ 2\\) 시 **+25점** 가점(옵션)
        """
    )
