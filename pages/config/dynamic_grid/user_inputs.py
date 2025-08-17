import streamlit as st
from decimal import Decimal
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType

from frontend.pages.config.utils import get_candles


def get_price_range_defaults(connector_name: str, trading_pair: str, interval: str, days: int = 7):
    """Fetch candles and compute default price range based on recent min/max prices."""
    try:
        candles = get_candles(
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval,
            days=days
        )
        current_price = float(candles['close'].iloc[-1])
        min_price = float(candles['low'].quantile(0.05))
        max_price = float(candles['high'].quantile(0.95))
        return round(min_price, 2), round(current_price, 2), round(max_price, 2)
    except Exception as e:
        st.warning(f"Could not fetch price data: {str(e)}. Using default values.")
        return 40000.0, 42000.0, 44000.0  # Fallback defaults


def user_inputs():
    # Split the page into two columns for the expanders
    left_col, right_col = st.columns(2)
    
    with left_col:
        # Basic Configuration
        with st.expander("基础配置", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                connector_name = st.text_input("连接器名称", value="binance_perpetual")
                trading_pair = st.text_input("交易对", value="WLD-USDT")
                leverage = st.number_input("杠杆", min_value=1, value=10)
            with c2:
                position_mode = st.selectbox(
                    "仓位模式",
                    options=["HEDGE", "ONEWAY"],
                    index=0
                )
                total_amount_quote = st.number_input(
                    "总投入资金 (USDT)",
                    min_value=0.0,
                    value=100.0,
                    help="总投入资金(USDT)"
                )
        
        # Dynamic Grid Configuration
        with st.expander("动态网格配置", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                grid_width_percentage = st.number_input(
                    "网格宽度百分比",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.03,
                    format="%.3f",
                    help="网格宽度百分比"
                )
                
                price_change_threshold = st.number_input(
                    "价格变化阈值",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.02,
                    format="%.3f",
                    help="价格变化阈值"
                )
            
            with c2:
                peak_detection_period = st.number_input(
                    "峰值检测周期(秒)",
                    min_value=60,
                    value=300,
                    help="峰值检测周期(秒)"
                )
                
                adjustment_interval = st.number_input(
                    "调整间隔(秒)",
                    min_value=60,
                    value=86400,
                    help="边界调整间隔(秒)"
                )

        # Trend Detection Configuration
        with st.expander("趋势判断配置", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                trend_lookback_periods = st.number_input(
                    "趋势回看周期数",
                    min_value=3,
                    max_value=20,
                    value=5,
                    help="趋势回看周期数"
                )
            
            with c2:
                trend_threshold = st.number_input(
                    "趋势判断阈值",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    format="%.3f",
                    help="趋势判断阈值"
                )
    
    with right_col:
        # Order Management Configuration
        with st.expander("订单管理配置", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                min_order_amount_quote = st.number_input(
                    "最小订单金额 (USDT)",
                    min_value=1.0,
                    value=5.0,
                    help="最小订单金额"
                )
                
                max_open_orders = st.number_input(
                    "最大同时开放订单数",
                    min_value=1,
                    value=3,
                    help="最大同时开放订单数"
                )
                activation_bounds = st.number_input(
                    "价格偏差触发更新",
                    min_value=0.0,
                    value=0.01,
                    format="%.4f",
                    help="价格偏差触发更新"
                )
            
            with c2:
                min_spread = st.number_input(
                    "订单之间的最小价差",
                    min_value=0.0000,
                    value=0.0001,
                    format="%.4f",
                    help="订单之间的最小价差",
                    step=0.0001
                )

                max_orders_per_batch = st.number_input(
                    "每批次最大订单数",
                    min_value=1,
                    value=1,
                    help="每批次最大订单数"
                )
                
                order_frequency = st.number_input(
                    "订单频率(秒)",
                    min_value=1,
                    value=3,
                    help="订单频率(秒)"
                )
                
                keep_position = st.checkbox(
                    "保持仓位",
                    value=False,
                    help="是否保持仓位"
                )
        
        # Triple Barrier Configuration
        with st.expander("三重障碍配置", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                # Order types
                open_order_type_options = ["LIMIT", "LIMIT_MAKER", "MARKET"]
                open_order_type = st.selectbox(
                    "开仓订单类型",
                    options=open_order_type_options,
                    index=1,  # Default to LIMIT_MAKER
                    key="open_order_type"
                )
                
                take_profit_order_type_options = ["LIMIT", "LIMIT_MAKER", "MARKET"]
                take_profit_order_type = st.selectbox(
                    "止盈订单类型",
                    options=take_profit_order_type_options,
                    index=1,  # Default to LIMIT_MAKER
                    key="tp_order_type"
                )
                
                stop_loss_order_type_options = ["LIMIT", "LIMIT_MAKER", "MARKET"]
                stop_loss_order_type = st.selectbox(
                    "止损订单类型",
                    options=stop_loss_order_type_options,
                    index=2,  # Default to MARKET
                    key="sl_order_type"
                )
            
            with c2:
                take_profit = st.number_input(
                    "止盈比例",
                    min_value=0.0,
                    value=grid_width_percentage,
                    format="%.4f",
                    help="止盈比例"
                )
                
                stop_loss = st.number_input(
                    "止损比例",
                    min_value=0.0,
                    value=grid_width_percentage*3,
                    format="%.4f",
                    help="止损比例"
                )
        
    # Chart Configuration
    with st.expander("图表配置", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            candles_connector = st.text_input(
                "K线连接器",
                value=connector_name,  # Use same connector as trading by default
                help="获取价格数据的连接器"
            )
        with c2:
            interval = st.selectbox(
                "时间间隔",
                options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"],
                index=5,  # Default to 1h
                help="K线时间间隔"
            )
        with c3:
            days_to_visualize = st.number_input(
                "显示天数",
                min_value=1,
                max_value=365,
                value=30,
                help="显示的历史数据天数"
            )
    
    # Prepare triple barrier config
    triple_barrier_config = {
        "open_order_type": OrderType[open_order_type],
        "take_profit_order_type": OrderType[take_profit_order_type],
        "stop_loss_order_type": OrderType[stop_loss_order_type],
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "time_limit": None,
    }
    
    return {
        "controller_name": "dynamic_grid",
        "controller_type": "generic",
        "connector_name": connector_name,
        "candles_connector": candles_connector,
        "trading_pair": trading_pair,
        "interval": interval,
        "days_to_visualize": days_to_visualize,
        "leverage": leverage,
        "position_mode": position_mode,
        "total_amount_quote": total_amount_quote,
        "grid_width_percentage": grid_width_percentage,
        "peak_detection_period": peak_detection_period,
        "price_change_threshold": price_change_threshold,
        "adjustment_interval": adjustment_interval,
        "trend_lookback_periods": trend_lookback_periods,
        "trend_threshold": trend_threshold,
        "min_order_amount_quote": min_order_amount_quote,
        "max_open_orders": max_open_orders,
        "max_orders_per_batch": max_orders_per_batch,
        "order_frequency": order_frequency,
        "keep_position": keep_position,
        "min_spread_between_orders": min_spread,
        "triple_barrier_config": triple_barrier_config,
        "activation_bounds": activation_bounds,
        "candles_config": []
    }