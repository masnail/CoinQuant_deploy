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
                connector_name = st.text_input("连接器名称", value="okx_perpetual")
                trading_pair = st.text_input("交易对", value="ETH-USDT-SWAP")
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
                avg_grid_width_percentage = st.number_input(
                    "单个网格宽度百分比",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.03,
                    format="%.3f",
                    help="单个网格宽度百分比"
                )

                all_grid_width_percentage = st.number_input(
                    "整个网格宽度百分比",
                    min_value=0.2,
                    max_value=1.0,
                    value=0.5,
                    format="%.3f",
                    help="整个网格宽度百分比"
                )
                
            
            with c2:
                exceed_bound_percentage = st.number_input(
                    "超出上下边界百分比",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.2,
                    format="%.3f",
                    help="超出上下边界百分比"
                )
                adjustment_interval = st.number_input(
                    "调整间隔(秒)",
                    min_value=300,
                    value=86400,
                    help="边界调整间隔(秒)"
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
                    value=avg_grid_width_percentage,
                    format="%.4f",
                    help="止盈比例"
                )
                
                stop_loss = st.number_input(
                    "止损比例",
                    min_value=0.0,
                    value=avg_grid_width_percentage*3,
                    format="%.4f",
                    help="止损比例"
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
        "id": "DynamicGrid",
        "controller_name": "dynamic_grid",
        "controller_type": "generic",
        "connector_name": connector_name,
        "trading_pair": trading_pair,
        "leverage": leverage,
        "position_mode": position_mode,
        "total_amount_quote": total_amount_quote,
        "avg_grid_width_percentage": avg_grid_width_percentage,
        "all_grid_width_percentage": all_grid_width_percentage,
        "exceed_bound_percentage": exceed_bound_percentage,
        "adjustment_interval": adjustment_interval,
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