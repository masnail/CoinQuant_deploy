import streamlit as st

from frontend.components.directional_trading_general_inputs import get_directional_trading_general_inputs
from frontend.components.risk_management import get_risk_management_inputs


def user_inputs():
    default_config = st.session_state.get("default_config", {})
    
    # connector_name, trading_pair, leverage, total_amount_quote, max_executors_per_side, cooldown_time, position_mode, \
    #     candles_connector_name, candles_trading_pair, interval = get_directional_trading_general_inputs()
    sl, tp, time_limit, ts_ap, ts_delta, take_profit_order_type = get_risk_management_inputs()
    
    with st.expander("General Settings", expanded=True):
        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
        with c1:
            connector_name = st.text_input("Connector", value="okx_perpetual")
        with c2:
            trading_pair = st.text_input("Trading Pair", value="ETH-USDT")
        with c3:
            leverage = st.number_input("Leverage", min_value=1, max_value=100, value=10)
        with c4:
            total_amount_quote = st.number_input("Total Amount Quote", min_value=1, max_value=10000, value=100)
        with c5:
            max_executors_per_side = st.number_input("Max Executors Per Side", min_value=1, max_value=10, value=3)
        with c6:
            cooldown_time = st.number_input("Cooldown Time (Mins)", min_value=1, max_value=1000, value=60)
        with c7:
            position_mode = st.selectbox("Position Mode", ["HEDGE", "ONEWAY"], index = 0)
        with c8:
            interval = st.selectbox("Interval", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"], index = 1)
            
    candles_connector_name = connector_name
    candles_trading_pair = trading_pair

    with st.expander("MA Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            ma1 = st.number_input("MA1", min_value=1, max_value=200, value=3)
        with c2:
            ma2 = st.number_input("MA2", min_value=1, max_value=200, value=5)
        with c3:
            ma3 = st.number_input("MA3", min_value=1, max_value=200, value=7)
            
    with st.expander("MACD Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            macd_fast = st.number_input("MACD Fast", min_value=1, max_value=200, value=12)
        with c2:
            macd_slow = st.number_input("MACD Slow", min_value=1, max_value=200, value=26)
        with c3:
            macd_signal = st.number_input("MACD Signal", min_value=1, max_value=200, value=9)
            
    with st.expander("RSI Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            rsi_period = st.number_input("RSI Period", min_value=1, max_value=200, value=14)
        with c2:
            rsi_upper = st.number_input("RSI Upper", min_value=1, max_value=200, value=70)
        with c3:
            rsi_lower = st.number_input("RSI Lower", min_value=1, max_value=200, value=30)
            
    with st.expander("EMA Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            ema1 = st.number_input("EMA1", min_value=1, max_value=200, value=3)
        with c2:
            ema2 = st.number_input("EMA2", min_value=1, max_value=200, value=5)
        with c3:
            ema3 = st.number_input("EMA3", min_value=1, max_value=200, value=7)
            
    with st.expander("ADX Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            adx_period = st.number_input("ADX Period", min_value=1, max_value=200, value=14)
        with c2:
            adx_threshold = st.number_input("ADX Threshold", min_value=1, max_value=200, value=70)
    
    return {
        "controller_name": "trend_following_v1",
        "controller_type": "directional_trading",
        "connector_name": connector_name,
        "trading_pair": trading_pair,
        "leverage": leverage,
        "total_amount_quote": total_amount_quote,
        "max_executors_per_side": max_executors_per_side,
        "cooldown_time": cooldown_time,
        "position_mode": position_mode,
        "candles_connector": candles_connector_name,
        "candles_trading_pair": candles_trading_pair,
        "interval": interval,
        "stop_loss": sl,
        "take_profit": tp,
        "time_limit": time_limit,
        "trailing_stop": {
            "activation_price": ts_ap,
            "trailing_delta": ts_delta
        },
        "take_profit_order_type": take_profit_order_type.value,
        
        # 自定义参数赋值
        "ma1": ma1,
        "ma2": ma2,
        "ma3": ma3,
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal": macd_signal,
        "rsi_period": rsi_period,
        "rsi_upper": rsi_upper,
        "rsi_lower": rsi_lower,
        "ema1": ema1,
        "ema2": ema2,
        "ema3": ema3,
        "adx_period": adx_period,
        "adx_threshold": adx_threshold,
    }
