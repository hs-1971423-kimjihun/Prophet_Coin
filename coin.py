import streamlit as st
from datetime import datetime, timezone, timedelta
import ccxt 
import pandas as pd 
import numpy as np
np.float_ = np.float64  # Prophet과 numpy가 호환이 안되서 임의로 맞춰줌
from prophet import Prophet  # Prophet 모듈에서 Prophet 클래스를 임포트
from pykrx import stock
from pykrx import bond
from streamlit_navigation_bar import st_navbar
import plotly.graph_objects as go  # Plotly에서 사용


######### 네비게이션 바 초기화 ################
pages = ["Future", "About"]
styles = {
    "nav": {
        "background-color": "#fff",
        "padding": "1rem 1.15rem",
        "display": "flex",
        "justify-content": "center",
        "box-shadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
        "border-bottom": "10px solid #ccc"  # 하단실선 적용안되는중
    },
    "div": {
        "width": "100%",
        "max-width": "1200px"
    },
    "ul": {
        "list-style": "none",
        "display": "flex",
        "align-items": "center",
        "padding": "0",
        "margin": "0"
    },
    "li": {
        "margin": "0 1rem"
    },
    "a": {
        "text-decoration": "none",
        "color": "#333",
        "font-weight": "500",
        "display": "flex",
        "align-items": "center"
    },
    "img": {
        "height": "40px",
        "width": "auto"
    },
    "span": {
        "font-size": "1rem",
        "transition": "color 0.3s ease"
    }
}


options = {
    "show_menu": False,
    "show_sidebar": False,
}

page = st_navbar(pages, styles=styles, options=options)


# About 페이지 로직
if page == "About":
    st.write("사용한 기술: streamlit, pandas, ccxt, prophet, pykrx")

# Future 페이지 로직
elif page == "Future":
    tab1, tab2 = st.tabs(["비트코인", "코스피200"])
    with tab1:
        
        # Binance에서 BTC/USDT의 OHLCV 데이터를 가져옵니다.
        binance = ccxt.binance()
        btc_ohlcv = binance.fetch_ohlcv("BTC/USDT")

        # 데이터를 DataFrame으로 변환하고 컬럼명을 지정합니다.
        df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

        # datetime 컬럼을 밀리초 단위에서 날짜-시간 형식으로 변환합니다.
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')

        # datetime 컬럼을 UTC에서 한국 시간(KST)으로 변환합니다.
        kst = timezone(timedelta(hours=9))  # 한국 시간대(KST) UTC+9
        df['datetime'] = df['datetime'].dt.tz_localize(timezone.utc).dt.tz_convert(kst)

        # 마지막 행의 datetime 값을 가져와서 출력 형식에 맞게 변환합니다.
        last_update = df['datetime'].iloc[-1].strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"최종 업데이트 날짜: {last_update}")

        # Prophet이 인식할 수 있도록 컬럼명을 'ds', 'y'로 변경하고 시간대 정보를 제거합니다.
        df['ds'] = df['datetime'].dt.tz_localize(None)  # 시간대 정보 제거
        df.rename(columns={'open': 'y'}, inplace=True)

        # Prophet 모델을 초기화하고 데이터를 학습시킵니다.
        m = Prophet()
        m.fit(df)

        # 향후 1시간(60분)의 데이터를 예측
        future = m.make_future_dataframe(periods=60, freq='T')  # '1min' 대신 'T' 사용
        forecast = m.predict(future)



         ############## 비트코인 현재 차트 그리기 ####################

        # 캔들차트 그리기
        fig2 = go.Figure(data=[go.Candlestick(x=df['datetime'],
                                              open=df['y'], 
                                              high=df['high'],
                                              low=df['low'],
                                              close=df['close'])])

        fig2.update_layout(
            title="비트코인 실시간 가격 차트 (BTC/USDT)",
            xaxis_title="날짜-시간",
            yaxis_title="가격 (USDT)",
            xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig2)


        ############################## 예측 데이터 출력 ##############################
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; ">
                <h4> AI의 데이터 예측</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        # 데이터 시각화
        fig1 = m.plot(forecast)

        # Streamlit에서 시각화된 그래프를 표시
        st.pyplot(fig1)

        ############# 상승추세인지 하락추세인지 판단 #####################

        # is_trend 변수 초기화
        is_trend = 0.0

        # 첫 번째 샘플 데이터 비교
        last_few_predictions1 = forecast['yhat'].tail(5)
        if last_few_predictions1.iloc[-1] > last_few_predictions1.iloc[0]:
            is_trend += 0.5  # 상승 추세
        else:
            is_trend -= 0.5  # 하락 추세

        # 두 번째 샘플 데이터 비교
        last_few_predictions2 = forecast['yhat'].tail(10)
        if last_few_predictions2.iloc[-1] > last_few_predictions2.iloc[0]:
            is_trend += 0.5  # 상승 추세
        else:
            is_trend -= 0.5  # 하락 추세

        # 최종 추세 판단
        trend_message = "미래에는 상승추세로 진행될 확률이 높습니다!" if is_trend > 0 else "미래에는 하락추세로 진행될 확률이 높아요."

        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; ">
                <h4>{trend_message}</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div style="display: flex; margin-top: 55px; margin-bottom: 300px;">
                해당 데이터는 단순 가격 데이터를 기준으로 예측한 데이터 이며, 투자의 책임은 본인에게 있습니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1,col2,col3 = st.columns([1,1,1])
        
        with col1 :
            # column 1 에 담을 내용
            st.image("./prophet.png", caption="Prophet")
        with col2 :
        # column 2 에 담을 내용
            st.image("./Python.png", caption="Python")
        with col3 :
        # column 3 에 담을 내용
            st.image("./ccxt.png", caption="ccxt")



    with tab2:
        st.write("코스피200 탭 내용")
        # 코스피200 관련 코드를 여기에 추가할 수 있습니다.
        df = stock.get_index_ohlcv("20240829", "20240830", "1028")
        st.write(df.head(2))



# for ticker in stock.get_market_ticker_list("2024-08-30"):
#         종목 = [ticker, stock.get_market_ticker_name(ticker)]
#         st.write(종목)

# #종목 코드
# item_code = "005930"
# url = "https://m.stock.naver.com/api/stock/%s/integration"%(item_code)
# #urllib.request를 통해 링크의 결과를 가져옵니다.
# raw_data = urllib.request.urlopen(url).read()
# #추후, 데이터 가공을 위해 json 형식으로 변경 합니다.
# json_data = json.loads(raw_data)


# st.write(json_data)

# #종목명 가져오기
# stock_name = json_data['stockName']
# print("종목명 : %s"%(stock_name))

# #가격 가져오기
# current_price = json_data['dealTrendInfos'][0]['closePrice']
# print("가격 : %s"%(current_price))

# #시총 가져오기
# for code in json_data['totalInfos']:
#     if 'marketValue' == code['code']:
#         marketSum_value = code['value']
#         print("시총 : %s"%(marketSum_value))

# #PER 가져오기
# for i in json_data['totalInfos']:
#     if 'per' == i['code']:
#         per_value_str = i['value']
#         print("PER : %s"%(per_value_str))


# #PBR 가져오기
# for v in json_data['totalInfos']:
#     if 'pbr' == v['code']:
#         pbr_value_str = v['value']
#         print("PBR : %s"%(pbr_value_str))

