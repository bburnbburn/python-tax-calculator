# app.py
import streamlit as st

st.set_page_config(page_title="소득세 계산기", layout="centered")
st.title("💰 소득 수준별 세금 계산기")

st.caption("단위: 만원 / 세율: 저 5%, 중 20%, 고 40%, 초고 50%")

# 입력 위젯
income = st.number_input("소득을 입력하세요 (만원)", min_value=0.0, step=10.0, format="%.1f")

def calc_tax(income: float):
    if income < 1000:
        rate, level = 0.05, "저소득층"
    elif income < 5000:
        rate, level = 0.20, "중간소득층"
    elif income < 10000:
        rate, level = 0.40, "고소득층"
    else:
        rate, level = 0.50, "초고소득층"
    tax = income * rate
    after = income - tax
    return level, rate, tax, after

if income > 0:
    level, rate, tax, after = calc_tax(income)

    st.subheader("결과")
    col1, col2, col3 = st.columns(3)
    col1.metric("소득 수준", level)
    col2.metric("세율", f"{rate*100:.0f}%")
    col3.metric("예상 세금", f"{tax:.1f} 만원")

    st.success(f"세후 소득: **{after:.1f} 만원**")

with st.expander("세율표 보기"):
    st.write(
        """
        - **저소득층** (< 1,000만원): 5%  
        - **중간소득층** (1,000~4,999만원): 20%  
        - **고소득층** (5,000~9,999만원): 40%  
        - **초고소득층** (10,000만원 이상): 50%
        """
    )
