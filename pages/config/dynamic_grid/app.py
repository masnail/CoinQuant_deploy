import streamlit as st
from decimal import Decimal

from frontend.components.config_loader import get_default_config_loader
from frontend.components.save_config import render_save_config
from frontend.pages.config.dynamic_grid.user_inputs import user_inputs
from frontend.pages.config.utils import get_candles
from frontend.st_utils import get_backend_api_client, initialize_st_page


def get_dynamic_grid_trace(current_price, grid_width_percentage):
    """Generate horizontal line traces for the dynamic grid boundaries."""
    traces = []
    
    if current_price is None:
        return traces
    
    # Calculate dynamic grid boundaries
    grid_width = float(current_price) * float(grid_width_percentage)
    upper_boundary = float(current_price) + grid_width / 2
    lower_boundary = float(current_price) - grid_width / 2
    
    # Current price line
    traces.append(go.Scatter(
        x=[],  # Will be set to full range when plotting
        y=[float(current_price), float(current_price)],
        mode='lines',
        line=dict(color='rgba(255, 255, 0, 1)', width=2, dash='solid'),
        name=f'Current Price: {float(current_price):,.2f}',
        hoverinfo='name'
    ))
    
    # Upper boundary line
    traces.append(go.Scatter(
        x=[],  # Will be set to full range when plotting
        y=[upper_boundary, upper_boundary],
        mode='lines',
        line=dict(color='rgba(0, 255, 0, 1)', width=1.5, dash='dot'),
        name=f'Upper Boundary: {upper_boundary:,.2f}',
        hoverinfo='name'
    ))
    
    # Lower boundary line
    traces.append(go.Scatter(
        x=[],  # Will be set to full range when plotting
        y=[lower_boundary, lower_boundary],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 1)', width=1.5, dash='dot'),
        name=f'Lower Boundary: {lower_boundary:,.2f}',
        hoverinfo='name'
    ))
    
    return traces

# Initialize the Streamlit page
initialize_st_page(title="动态网格策略配置", icon="🔄", initial_sidebar_state="expanded")
backend_api_client = get_backend_api_client()

get_default_config_loader("dynamic_grid")

# User inputs
inputs = user_inputs()
st.session_state["default_config"].update(inputs)

# Strategy description
st.markdown("### 策略说明")
st.markdown("""
**动态网格策略特点:**

1. **自动边界初始化**: 基于当前市场价格自动计算初始网格边界

2. **峰值检测**: 自动识别价格的峰值和谷值

3. **动态边界调整**: 动态调整网格上下边界

4. **趋势判断**: 
   - 价格上穿网格上边界时开多（做多）
   - 价格下穿网格下边界时开空（做空）
   - 价格在网格内部时不进行交易，等待穿越信号

5. **智能网格重启**: 在价格突破边界或趋势发生重大变化时重启网格

""")

# Configuration save section
def prepare_config_for_save(config):
    """Prepare the configuration for saving by converting to proper format."""
    prepared_config = config.copy()
    
    # Convert triple barrier order types to values
    if "triple_barrier_config" in prepared_config and prepared_config["triple_barrier_config"]:
        for key in ["open_order_type", "stop_loss_order_type", "take_profit_order_type"]:
            if key in prepared_config["triple_barrier_config"] and prepared_config["triple_barrier_config"][key] is not None:
                prepared_config["triple_barrier_config"][key] = prepared_config["triple_barrier_config"][key].value
    
    # Remove visualization-only fields
    fields_to_remove = ['candles_connector', 'interval', 'days_to_visualize']
    for field in fields_to_remove:
        prepared_config.pop(field, None)
    
    return prepared_config

render_save_config(
    st.session_state["default_config"]["id"],
    prepare_config_for_save(st.session_state["default_config"])
)