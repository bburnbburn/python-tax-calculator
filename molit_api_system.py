import streamlit as st
import pandas as pd
import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
import time

# 페이지 설정
st.set_page_config(
    page_title="전국 실거래가 이상거래 탐지 시스템",
    page_icon="🏠",
    layout="wide"
)

# 전국 시도 및 시군구 코드
REGION_CODES = {
    '서울특별시': {
        '전체': ['11110', '11140', '11170', '11200', '11215', '11230', '11260', '11290', '11305', '11320', '11350', '11380', '11410', '11440', '11470', '11500', '11530', '11545', '11560', '11590', '11620', '11650', '11680', '11710', '11740'],
        '종로구': ['11110'], '중구': ['11140'], '용산구': ['11170'], '성동구': ['11200'], '광진구': ['11215'],
        '동대문구': ['11230'], '중랑구': ['11260'], '성북구': ['11290'], '강북구': ['11305'], '도봉구': ['11320'],
        '노원구': ['11350'], '은평구': ['11380'], '서대문구': ['11410'], '마포구': ['11440'], '양천구': ['11470'],
        '강서구': ['11500'], '구로구': ['11530'], '금천구': ['11545'], '영등포구': ['11560'], '동작구': ['11590'],
        '관악구': ['11620'], '서초구': ['11650'], '강남구': ['11680'], '송파구': ['11710'], '강동구': ['11740']
    },
    '부산광역시': {
        '전체': ['26110', '26140', '26170', '26200', '26230', '26260', '26290', '26320', '26350', '26380', '26410', '26440', '26470', '26500', '26530', '26710'],
        '중구': ['26110'], '서구': ['26140'], '동구': ['26170'], '영도구': ['26200'], '부산진구': ['26230'],
        '동래구': ['26260'], '남구': ['26290'], '북구': ['26320'], '해운대구': ['26350'], '사하구': ['26380'],
        '금정구': ['26410'], '강서구': ['26440'], '연제구': ['26470'], '수영구': ['26500'], '사상구': ['26530'], '기장군': ['26710']
    },
    '대구광역시': {
        '전체': ['27110', '27140', '27170', '27200', '27230', '27260', '27290', '27710'],
        '중구': ['27110'], '동구': ['27140'], '서구': ['27170'], '남구': ['27200'],
        '북구': ['27230'], '수성구': ['27260'], '달서구': ['27290'], '달성군': ['27710']
    },
    '인천광역시': {
        '전체': ['28110', '28140', '28177', '28185', '28200', '28237', '28245', '28260', '28710', '28720'],
        '중구': ['28110'], '동구': ['28140'], '미추홀구': ['28177'], '연수구': ['28185'], '남동구': ['28200'],
        '부평구': ['28237'], '계양구': ['28245'], '서구': ['28260'], '강화군': ['28710'], '옹진군': ['28720']
    },
    '광주광역시': {
        '전체': ['29110', '29140', '29155', '29170', '29200'],
        '동구': ['29110'], '서구': ['29140'], '남구': ['29155'], '북구': ['29170'], '광산구': ['29200']
    },
    '대전광역시': {
        '전체': ['30110', '30140', '30170', '30200', '30230'],
        '동구': ['30110'], '중구': ['30140'], '서구': ['30170'], '유성구': ['30200'], '대덕구': ['30230']
    },
    '울산광역시': {
        '전체': ['31110', '31140', '31170', '31200', '31710'],
        '중구': ['31110'], '남구': ['31140'], '동구': ['31170'], '북구': ['31200'], '울주군': ['31710']
    },
    '세종특별자치시': {
        '전체': ['36110'],
        '세종시': ['36110']
    },
    '경기도': {
        '전체': ['41111', '41113', '41115', '41117', '41131', '41133', '41135', '41150', '41171', '41173', '41190', '41210', '41220', '41250', '41271', '41273', '41281', '41285', '41287', '41290', '41310', '41360', '41370', '41390', '41410', '41430', '41450', '41461', '41463', '41465', '41480', '41500', '41550', '41570', '41590', '41610', '41630', '41650', '41670', '41800', '41820', '41830'],
        '수원시': ['41111', '41113', '41115', '41117'], '성남시': ['41131', '41133', '41135'], '의정부시': ['41150'],
        '안양시': ['41171', '41173'], '부천시': ['41190'], '광명시': ['41210'], '평택시': ['41220'], '동두천시': ['41250'],
        '안산시': ['41271', '41273'], '고양시': ['41281', '41285', '41287'], '과천시': ['41290'], '구리시': ['41310'],
        '남양주시': ['41360'], '오산시': ['41370'], '시흥시': ['41390'], '군포시': ['41410'], '의왕시': ['41430'],
        '하남시': ['41450'], '용인시': ['41461', '41463', '41465'], '파주시': ['41480'], '이천시': ['41500'],
        '안성시': ['41550'], '김포시': ['41570'], '화성시': ['41590'], '광주시': ['41610'], '양주시': ['41630'],
        '포천시': ['41650'], '여주시': ['41670'], '연천군': ['41800'], '가평군': ['41820'], '양평군': ['41830']
    },
    '강원특별자치도': {
        '전체': ['42110', '42130', '42150', '42170', '42190', '42210', '42230', '42720', '42730', '42750', '42760', '42770', '42780', '42790', '42800', '42810', '42820', '42830'],
        '춘천시': ['42110'], '원주시': ['42130'], '강릉시': ['42150'], '동해시': ['42170'], '태백시': ['42190'],
        '속초시': ['42210'], '삼척시': ['42230'], '홍천군': ['42720'], '횡성군': ['42730'], '영월군': ['42750'],
        '평창군': ['42760'], '정선군': ['42770'], '철원군': ['42780'], '화천군': ['42790'], '양구군': ['42800'],
        '인제군': ['42810'], '고성군': ['42820'], '양양군': ['42830']
    },
    '충청북도': {
        '전체': ['43111', '43112', '43113', '43114', '43130', '43150', '43720', '43730', '43740', '43745', '43750', '43760', '43770', '43800'],
        '청주시': ['43111', '43112', '43113', '43114'], '충주시': ['43130'], '제천시': ['43150'],
        '보은군': ['43720'], '옥천군': ['43730'], '영동군': ['43740'], '증평군': ['43745'], '진천군': ['43750'],
        '괴산군': ['43760'], '음성군': ['43770'], '단양군': ['43800']
    },
    '충청남도': {
        '전체': ['44131', '44133', '44150', '44180', '44200', '44210', '44230', '44250', '44270', '44710', '44760', '44770', '44790', '44800', '44810', '44825'],
        '천안시': ['44131', '44133'], '공주시': ['44150'], '보령시': ['44180'], '아산시': ['44200'], '서산시': ['44210'],
        '논산시': ['44230'], '계룡시': ['44250'], '당진시': ['44270'], '금산군': ['44710'], '부여군': ['44760'],
        '서천군': ['44770'], '청양군': ['44790'], '홍성군': ['44800'], '예산군': ['44810'], '태안군': ['44825']
    },
    '전라북도': {
        '전체': ['45111', '45113', '45130', '45140', '45180', '45190', '45210', '45710', '45720', '45730', '45740', '45750', '45770', '45790', '45800'],
        '전주시': ['45111', '45113'], '군산시': ['45130'], '익산시': ['45140'], '정읍시': ['45180'], '남원시': ['45190'],
        '김제시': ['45210'], '완주군': ['45710'], '진안군': ['45720'], '무주군': ['45730'], '장수군': ['45740'],
        '임실군': ['45750'], '순창군': ['45770'], '고창군': ['45790'], '부안군': ['45800']
    },
    '전라남도': {
        '전체': ['46110', '46130', '46150', '46170', '46230', '46710', '46720', '46730', '46770', '46780', '46790', '46800', '46810', '46820', '46830', '46840', '46860', '46870', '46880', '46890', '46900', '46910'],
        '목포시': ['46110'], '여수시': ['46130'], '순천시': ['46150'], '나주시': ['46170'], '광양시': ['46230'],
        '담양군': ['46710'], '곡성군': ['46720'], '구례군': ['46730'], '고흥군': ['46770'], '보성군': ['46780'],
        '화순군': ['46790'], '장흥군': ['46800'], '강진군': ['46810'], '해남군': ['46820'], '영암군': ['46830'],
        '무안군': ['46840'], '함평군': ['46860'], '영광군': ['46870'], '장성군': ['46880'], '완도군': ['46890'],
        '진도군': ['46900'], '신안군': ['46910']
    },
    '경상북도': {
        '전체': ['47111', '47113', '47130', '47150', '47170', '47190', '47210', '47230', '47250', '47280', '47290', '47720', '47730', '47750', '47760', '47770', '47820', '47830', '47840', '47850', '47900', '47920', '47930', '47940'],
        '포항시': ['47111', '47113'], '경주시': ['47130'], '김천시': ['47150'], '안동시': ['47170'], '구미시': ['47190'],
        '영주시': ['47210'], '영천시': ['47230'], '상주시': ['47250'], '문경시': ['47280'], '경산시': ['47290'],
        '군위군': ['47720'], '의성군': ['47730'], '청송군': ['47750'], '영양군': ['47760'], '영덕군': ['47770'],
        '청도군': ['47820'], '고령군': ['47830'], '성주군': ['47840'], '칠곡군': ['47850'], '예천군': ['47900'],
        '봉화군': ['47920'], '울진군': ['47930'], '울릉군': ['47940']
    },
    '경상남도': {
        '전체': ['48121', '48123', '48125', '48127', '48129', '48170', '48220', '48240', '48250', '48270', '48310', '48330', '48720', '48730', '48740', '48820', '48840', '48850', '48860', '48870', '48880', '48890'],
        '창원시': ['48121', '48123', '48125', '48127', '48129'], '진주시': ['48170'], '통영시': ['48220'], '사천시': ['48240'],
        '김해시': ['48250'], '밀양시': ['48270'], '거제시': ['48310'], '양산시': ['48330'], '의령군': ['48720'],
        '함안군': ['48730'], '창녕군': ['48740'], '고성군': ['48820'], '남해군': ['48840'], '하동군': ['48850'],
        '산청군': ['48860'], '함양군': ['48870'], '거창군': ['48880'], '합천군': ['48890']
    },
    '제주특별자치도': {
        '전체': ['50110', '50130'],
        '제주시': ['50110'], '서귀포시': ['50130']
    }
}

class MOLITAPIClient:
    """국토부 실거래가 API 클라이언트"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev"
    
    def fetch_apt_trade(self, lawd_cd, deal_ymd):
        """아파트 실거래가 조회"""
        params = {
            'serviceKey': self.api_key,
            'LAWD_CD': lawd_cd,
            'DEAL_YMD': deal_ymd,
            'numOfRows': '1000'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                return self.parse_xml(response.content)
            else:
                return []
        except Exception as e:
            return []
    
    def parse_xml(self, xml_content):
        """XML 파싱"""
        try:
            root = ET.fromstring(xml_content)
            
            # 결과 코드 확인
            result_code = root.findtext('.//resultCode')
            if result_code != '00':
                return []
            
            items = []
            for item in root.findall('.//item'):
                try:
                    # 거래금액 전처리 (공백, 쉼표 제거)
                    price_str = item.findtext('거래금액', '0').strip().replace(',', '')
                    
                    data = {
                        '지역코드': item.findtext('지역코드'),
                        '시군구': item.findtext('법정동'),
                        '아파트': item.findtext('아파트'),
                        '거래금액': int(price_str) if price_str.isdigit() else 0,
                        '건축년도': int(item.findtext('건축년도', '0')),
                        '년': int(item.findtext('년', '0')),
                        '월': int(item.findtext('월', '0')),
                        '일': int(item.findtext('일', '0')),
                        '전용면적': float(item.findtext('전용면적', '0')),
                        '층': int(item.findtext('층', '0')) if item.findtext('층', '0').replace('-','').isdigit() else 0,
                        '도로명': item.findtext('도로명'),
                        '해제사유발생일': item.findtext('해제사유발생일'),
                        '거래유형': item.findtext('거래유형', '직거래'),
                    }
                    items.append(data)
                except Exception as e:
                    continue
            
            return items
            
        except ET.ParseError as e:
            return []

def analyze_anomalies(df):
    """이상거래 분석"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 평수 계산
    df['평수'] = (df['전용면적'] * 0.3025).round(1)
    
    # 평당가 계산
    df['평당가'] = (df['거래금액'] / df['평수']).round(0)
    
    # 단지별 평균 계산
    df['단지평균가'] = df.groupby(['아파트', '전용면적'])['거래금액'].transform('mean')
    df['단지평균평당가'] = df.groupby(['아파트', '전용면적'])['평당가'].transform('mean')
    
    # 가격 편차율
    df['가격편차율'] = ((df['거래금액'] - df['단지평균가']) / df['단지평균가'] * 100).round(1)
    
    # 평당가 편차율
    df['평당가편차율'] = ((df['평당가'] - df['단지평균평당가']) / df['단지평균평당가'] * 100).round(1)
    
    # 이상 신호 점수 계산
    def calculate_anomaly_score(row):
        score = 0
        reasons = []
        
        # 1. 급격한 가격 상승 (30% 이상)
        if row['가격편차율'] > 30:
            score += 40
            reasons.append('급격한 가격 상승')
        elif row['가격편차율'] > 20:
            score += 25
            reasons.append('가격 상승')
        
        # 2. 급격한 가격 하락 (20% 이상)
        if row['가격편차율'] < -20:
            score += 35
            reasons.append('급격한 가격 하락')
        elif row['가격편차율'] < -15:
            score += 20
            reasons.append('가격 하락')
        
        # 3. 평당가 이상
        if abs(row['평당가편차율']) > 25:
            score += 20
            reasons.append('평당가 이상')
        
        # 4. 저층 고가 (5층 이하인데 평균보다 높음)
        if row['층'] <= 5 and row['가격편차율'] > 10:
            score += 15
            reasons.append('저층 고가')
        
        # 5. 해제사유 있음
        if pd.notna(row['해제사유발생일']) and row['해제사유발생일'] != '':
            score += 30
            reasons.append('거래 해제 이력')
        
        # 6. 초고층 저가 (20층 이상인데 평균보다 낮음)
        if row['층'] >= 20 and row['가격편차율'] < -10:
            score += 10
            reasons.append('고층 저가')
        
        return score, reasons
    
    df[['이상점수', '이상사유']] = df.apply(
        lambda row: pd.Series(calculate_anomaly_score(row)), 
        axis=1
    )
    
    # 이상거래 여부
    df['이상거래'] = df['이상점수'] >= 20
    
    return df

def main():
    st.title("🏠 전국 실거래가 이상거래 탐지 시스템")
    st.markdown("**국토부 실거래가 Open API 실시간 연동 (전국 17개 시도)**")
    st.markdown("---")
    
    # 사이드바 - API 설정
    with st.sidebar:
        st.header("⚙️ API 설정")
        
        # API 키 입력
        api_key = st.text_input(
            "국토부 API 인증키",
            type="password",
            help="공공데이터포털(data.go.kr)에서 발급받은 인증키를 입력하세요"
        )
        
        if not api_key:
            st.warning("⚠️ API 인증키를 입력해주세요")
            st.markdown("""
            **API 키 발급 방법:**
            1. [공공데이터포털](https://www.data.go.kr) 접속
            2. 회원가입 및 로그인
            3. '아파트매매 실거래 상세 자료' 검색
            4. 활용신청 → 인증키 발급
            5. 승인까지 1-2시간 소요
            """)
        
        st.markdown("---")
        
        # 검색 조건
        st.header("🔍 검색 조건")
        
        # 시도 선택
        selected_sido = st.selectbox(
            "시/도 선택",
            list(REGION_CODES.keys())
        )
        
        # 시군구 선택
        sigungu_options = list(REGION_CODES[selected_sido].keys())
        selected_sigungu = st.selectbox(
            "시/군/구 선택",
            sigungu_options
        )
        
        # 기간 선택
        search_date = st.date_input(
            "조회 년월",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
        
        deal_ymd = search_date.strftime('%Y%m')
        
        # 이상거래 임계값
        st.markdown("---")
        st.subheader("📊 탐지 설정")
        min_score = st.slider(
            "최소 이상점수",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="이 점수 이상인 거래만 표시"
        )
        
        # 최대 조회 건수
        max_codes = st.slider(
            "최대 조회 지역 수",
            min_value=1,
            max_value=50,
            value=10,
            help="'전체' 선택 시 조회할 최대 지역 수 (API 제한 고려)"
        )
        
        # 데이터 조회 버튼
        st.markdown("---")
        search_button = st.button("🔎 데이터 조회", type="primary", use_container_width=True)
        
        # 안내
        if selected_sigungu == '전체':
            st.info(f"💡 '{selected_sido}' 전체 지역을 조회합니다. 시간이 소요될 수 있습니다.")
    
    # API 키가 없으면 안내 표시
    if not api_key:
        st.info("👈 왼쪽 사이드바에서 API 인증키를 입력하고 데이터를 조회하세요")
        
        # 지원 지역 안내
        st.subheader("📍 지원 지역")
        cols = st.columns(3)
        for idx, sido in enumerate(list(REGION_CODES.keys())):
            with cols[idx % 3]:
                st.write(f"✅ **{sido}**")
                st.caption(f"{len(REGION_CODES[sido])-1}개 시군구")
        
        return
    
    # 데이터 조회
    if search_button:
        with st.spinner(f'🔄 {selected_sido} {selected_sigungu} 실거래가 데이터 조회 중...'):
            client = MOLITAPIClient(api_key)
            
            all_data = []
            region_codes = REGION_CODES[selected_sido][selected_sigungu]
            
            # 전체 선택 시 최대 조회 수 제한
            if selected_sigungu == '전체' and len(region_codes) > max_codes:
                region_codes = region_codes[:max_codes]
                st.warning(f"⚠️ API 제한으로 인해 {max_codes}개 지역만 조회합니다.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, code in enumerate(region_codes):
                status_text.text(f"지역코드 {code} 조회 중... ({idx+1}/{len(region_codes)})")
                
                data = client.fetch_apt_trade(code, deal_ymd)
                all_data.extend(data)
                
                progress_bar.progress((idx + 1) / len(region_codes))
                time.sleep(0.3)  # API 요청 간격
            
            progress_bar.empty()
            status_text.empty()
            
            if not all_data:
                st.warning("⚠️ 조회된 데이터가 없습니다. API 키와 조회 조건을 확인해주세요.")
                return
            
            # 데이터프레임 생성
            df = pd.DataFrame(all_data)
            
            # 이상거래 분석
            with st.spinner('📊 이상거래 분석 중...'):
                df_analyzed = analyze_anomalies(df)
            
            # 세션에 저장
            st.session_state['df_analyzed'] = df_analyzed
            st.session_state['search_region'] = f"{selected_sido} {selected_sigungu}"
            st.session_state['search_date'] = deal_ymd
            
            st.success(f"✅ 총 {len(df):,} 건의 실거래 데이터를 조회했습니다!")
    
    # 분석 결과 표시
    if 'df_analyzed' in st.session_state:
        df_analyzed = st.session_state['df_analyzed']
        
        # 필터링
        df_filtered = df_analyzed[df_analyzed['이상점수'] >= min_score].copy()
        df_filtered = df_filtered.sort_values('이상점수', ascending=False)
        
        # 통계 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("전체 거래", f"{len(df_analyzed):,}건")
        
        with col2:
            anomaly_count = df_analyzed['이상거래'].sum()
            anomaly_rate = (anomaly_count / len(df_analyzed) * 100) if len(df_analyzed) > 0 else 0
            st.metric("이상거래", f"{anomaly_count:,}건", f"{anomaly_rate:.1f}%")
        
        with col3:
            high_price = (df_analyzed['가격편차율'] > 20).sum()
            st.metric("급등 의심", f"{high_price:,}건")
        
        with col4:
            low_price = (df_analyzed['가격편차율'] < -20).sum()
            st.metric("급락 의심", f"{low_price:,}건")
        
        st.markdown("---")
        
        # 지역 정보 표시
        st.info(f"📍 **조회 지역**: {st.session_state['search_region']} | **조회 기간**: {st.session_state['search_date']}")
        
        # 상위 이상거래 TOP 10
        st.subheader("🏆 이상점수 TOP 10")
        top_10 = df_filtered.head(10)
        
        if len(top_10) > 0:
            for idx, row in top_10.iterrows():
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                
                with col1:
                    if row['이상점수'] >= 40:
                        st.error(f"**{row['이상점수']}점**")
                    elif row['이상점수'] >= 25:
                        st.warning(f"**{row['이상점수']}점**")
                    else:
                        st.info(f"**{row['이상점수']}점**")
                
                with col2:
                    st.write(f"**{row['아파트']}**")
                    st.caption(f"{row['시군구']}")
                
                with col3:
                    st.write(f"{row['거래금액']:,}만원")
                    st.caption(f"{row['전용면적']:.1f}㎡ ({row['평수']:.1f}평)")
                
                with col4:
                    if row['가격편차율'] > 0:
                        st.write(f"🔺 +{row['가격편차율']:.1f}%")
                    else:
                        st.write(f"🔻 {row['가격편차율']:.1f}%")
                    st.caption(' / '.join(row['이상사유'][:2]))
        else:
            st.info("조건에 맞는 이상거래가 없습니다.")
        
        st.markdown("---")
        
        # 이상거래 목록
        st.subheader(f"🚨 이상거래 탐지 결과 ({len(df_filtered):,}건)")
        
        # 필터 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_anomaly_type = st.selectbox(
                "이상 유형",
                ['전체', '급등 의심', '급락 의심', '해제 이력']
            )
        
        with col2:
            min_price = st.number_input(
                "최소 거래금액 (만원)",
                min_value=0,
                value=0,
                step=1000
            )
        
        with col3:
            min_area = st.number_input(
                "최소 면적 (㎡)",
                min_value=0.0,
                value=0.0,
                step=10.0
            )
        
        # 추가 필터링
        if filter_anomaly_type == '급등 의심':
            df_filtered = df_filtered[df_filtered['가격편차율'] > 20]
        elif filter_anomaly_type == '급락 의심':
            df_filtered = df_filtered[df_filtered['가격편차율'] < -20]
        elif filter_anomaly_type == '해제 이력':
            df_filtered = df_filtered[pd.notna(df_filtered['해제사유발생일']) & (df_filtered['해제사유발생일'] != '')]
        
        if min_price > 0:
            df_filtered = df_filtered[df_filtered['거래금액'] >= min_price]
        
        if min_area > 0:
            df_filtered = df_filtered[df_filtered['전용면적'] >= min_area]
        
        st.write(f"**필터링 결과: {len(df_filtered):,}건**")
        
        if len(df_filtered) == 0:
            st.info("조건에 맞는 이상거래가 없습니다. 필터 조건을 변경해보세요.")
        else:
            # 페이지네이션
            items_per_page = 20
            total_pages = (len(df_filtered) - 1) // items_per_page + 1
            
            page = st.selectbox(
                f"페이지 선택 (총 {total_pages}페이지)",
                range(1, total_pages + 1)
            )
            
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(df_filtered))
            
            df_page = df_filtered.iloc[start_idx:end_idx]
            
            for idx, row in df_page.iterrows():
                # 위험도에 따른 색상
                if row['이상점수'] >= 40:
                    color = "🔴"
                elif row['이상점수'] >= 25:
                    color = "🟠"
                else:
                    color = "🟡"
                
                with st.expander(
                    f"{color} **{row['아파트']}** | {row['시군구']} | "
                    f"점수: {row['이상점수']}점 | 편차: {row['가격편차율']:+.1f}%"
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**📍 기본 정보**")
                        st.write(f"- 지역: {row['시군구']}")
                        st.write(f"- 단지: {row['아파트']}")
                        st.write(f"- 층: {row['층']}층")
                        st.write(f"- 건축년도: {row['건축년도']}년")
                        st.write(f"- 전용면적: {row['전용면적']:.2f}㎡ ({row['평수']:.1f}평)")
                        if row['도로명']:
                            st.write(f"- 도로명: {row['도로명']}")
                    
                    with col2:
                        st.markdown("**💰 가격 정보**")
                        st.write(f"- 거래금액: **{row['거래금액']:,}만원**")
                        st.write(f"- 단지평균: {row['단지평균가']:,.0f}만원")
                        st.write(f"- 가격편차: **{row['가격편차율']:+.1f}%**")
                        st.write(f"- 평당가: {row['평당가']:,.0f}만원")
                        st.write(f"- 평균평당가: {row['단지평균평당가']:,.0f}만원")
                        st.write(f"- 거래유형: {row['거래유형']}")
                    
                    with col3:
                        st.markdown("**🔍 이상 분석**")
                        st.write(f"- 이상점수: **{row['이상점수']}점**")
                        st.write(f"- 거래일: {row['년']}-{row['월']:02d}-{row['일']:02d}")
                        
                        if row['이상사유']:
                            st.write("- 이상사유:")
                            for reason in row['이상사유']:
                                st.write(f"  • {reason}")
                        
                        if pd.notna(row['해제사유발생일']) and row['해제사유발생일'] != '':
                            st.error(f"⚠️ 거래 해제: {row['해제사유발생일']}")
        
        st.markdown("---")
        
        # 통계 분석
        st.subheader("📊 통계 분석")
        
        tab1, tab2, tab3 = st.tabs(["지역별 분석", "가격대별 분석", "면적별 분석"])
        
        with tab1:
            # 지역별 이상거래 건수
            region_stats = df_analyzed[df_analyzed['이상거래']].groupby('시군구').size().sort_values(ascending=False).head(10)
            
            if len(region_stats) > 0:
                st.write("**이상거래가 많은 지역 TOP 10**")
                for region, count in region_stats.items():
                    st.write(f"- {region}: {count}건")
            else:
                st.info("이상거래가 탐지되지 않았습니다.")
        
        with tab2:
            # 가격대별 분석
            df_analyzed['가격대'] = pd.cut(
                df_analyzed['거래금액'],
                bins=[0, 30000, 50000, 70000, 100000, float('inf')],
                labels=['3억 이하', '3~5억', '5~7억', '7~10억', '10억 이상']
            )
            
            price_stats = df_analyzed[df_analyzed['이상거래']].groupby('가격대').size()
            
            st.write("**가격대별 이상거래 분포**")
            for price_range, count in price_stats.items():
                st.write(f"- {price_range}: {count}건")
        
        with tab3:
            # 면적별 분석
            df_analyzed['면적대'] = pd.cut(
                df_analyzed['전용면적'],
                bins=[0, 60, 85, 100, 135, float('inf')],
                labels=['소형 (60㎡ 이하)', '중소형 (60~85㎡)', '중형 (85~100㎡)', '중대형 (100~135㎡)', '대형 (135㎡ 이상)']
            )
            
            area_stats = df_analyzed[df_analyzed['이상거래']].groupby('면적대').size()
            
            st.write("**면적대별 이상거래 분포**")
            for area_range, count in area_stats.items():
                st.write(f"- {area_range}: {count}건")
        
        st.markdown("---")
        
        # 데이터 다운로드
        st.subheader("📥 데이터 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 전체 데이터
            csv_all = df_analyzed.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📊 전체 데이터 다운로드 (CSV)",
                data=csv_all,
                file_name=f"trade_all_{st.session_state['search_date']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # 이상거래만
            csv_anomaly = df_filtered.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="🚨 이상거래만 다운로드 (CSV)",
                data=csv_anomaly,
                file_name=f"trade_anomaly_{st.session_state['search_date']}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # 안내 정보
        st.markdown("---")
        st.info("""
        **📌 이상거래 탐지 알고리즘**
        - **급격한 가격 상승**: 단지 평균 대비 30% 이상 높은 거래 (40점)
        - **급격한 가격 하락**: 단지 평균 대비 20% 이상 낮은 거래 (35점)
        - **평당가 이상**: 평당 가격이 평균 대비 25% 이상 차이 (20점)
        - **거래 해제 이력**: 해제사유발생일이 있는 거래 (30점)
        - **저층 고가**: 5층 이하인데 평균보다 10% 이상 높음 (15점)
        - **고층 저가**: 20층 이상인데 평균보다 10% 이상 낮음 (10점)
        
        ⚠️ **주의사항**
        - 이 시스템은 통계적 분석을 기반으로 하며, 실제 이상거래 여부는 추가 조사가 필요합니다.
        - 신축, 재건축, 리모델링 등 정상적인 가격 변동도 이상거래로 탐지될 수 있습니다.
        - API 호출 제한으로 인해 한 번에 많은 지역을 조회하면 시간이 소요됩니다.
        """)

if __name__ == "__main__":
    main()