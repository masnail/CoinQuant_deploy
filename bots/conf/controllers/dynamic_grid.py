import asyncio
from decimal import Decimal
from typing import List, Optional
import time

from pydantic import Field

from hummingbot.core.data_type.common import MarketDict, OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class DynamicGridConfig(ControllerConfigBase):
    """
    动态网格策略配置类
    根据价格峰值动态调整网格上下限，在上涨时开多，下跌时开空
    """
    controller_type: str = "generic"
    controller_name: str = "dynamic_grid"
    candles_config: List[CandlesConfig] = []

    # 账户配置
    leverage: int = 20
    position_mode: PositionMode = PositionMode.HEDGE

    # 基础交易配置
    connector_name: str = "binance_perpetual"
    trading_pair: str = "WLD-USDT"
    
    # 动态边界配置
    grid_width_percentage: Decimal = Field(default=Decimal("0.1"), json_schema_extra={"is_updatable": True})  # 网格宽度百分比
    
    # 峰值检测配置
    peak_detection_period: int = Field(default=300, json_schema_extra={"is_updatable": True})  # 峰值检测周期(秒)
    price_change_threshold: Decimal = Field(default=Decimal("0.02"), json_schema_extra={"is_updatable": True})  # 价格变化阈值
    adjustment_interval: int = Field(default=86400, json_schema_extra={"is_updatable": True})  # 调整间隔(秒)
    
    # 注意：使用adjustment_interval作为统一的调整间隔 / Note: Using adjustment_interval as unified adjustment interval
    
    # 趋势判断配置
    trend_lookback_periods: int = Field(default=5, json_schema_extra={"is_updatable": True})  # 趋势回看周期数
    trend_threshold: Decimal = Field(default=Decimal("0.01"), json_schema_extra={"is_updatable": True})  # 趋势判断阈值

    # 资金配置
    total_amount_quote: Decimal = Field(default=Decimal("1000"), json_schema_extra={"is_updatable": True})

    min_order_amount_quote: Optional[Decimal] = Field(default=Decimal("5"), json_schema_extra={"is_updatable": True})

    # 执行配置
    max_open_orders: int = Field(default=3, json_schema_extra={"is_updatable": True})
    max_orders_per_batch: Optional[int] = Field(default=1, json_schema_extra={"is_updatable": True})
    order_frequency: int = Field(default=5, json_schema_extra={"is_updatable": True})

    keep_position: bool = Field(default=False, json_schema_extra={"is_updatable": True})

    # 风险管理
    triple_barrier_config: TripleBarrierConfig = TripleBarrierConfig(
        take_profit=Decimal("0.002"),
        stop_loss=Decimal("0.01"),
        open_order_type=OrderType.LIMIT_MAKER,
        take_profit_order_type=OrderType.LIMIT_MAKER,
        stop_loss_order_type=OrderType.MARKET,
    )

    def update_markets(self, markets: MarketDict) -> MarketDict:
        return markets.add_or_update(self.connector_name, self.trading_pair)


class DynamicGrid(ControllerBase):
    def __init__(self, config: DynamicGridConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        
        # 动态边界状态 - 将在首次获取市场价格时初始化
        self.current_start_price = None
        self.current_end_price = None
        self.current_side = None  # 当前交易方向，初始为None（不操作）
        self.is_initialized = False  # 网格是否已初始化
        
        # 价格精度自动检测
        self.price_precision = None  # 自动检测的价格精度
        
        # 峰值检测状态
        self.price_history = []  # 价格历史记录
        self.last_peak_price = Decimal("0")
        self.last_valley_price = Decimal("999999")
        self.last_adjustment_time = 0
        
        # 趋势状态
        self.trend_direction = "neutral"  # "up", "down", "neutral"
        self.trend_prices = []  # 趋势判断价格序列
        self.last_trend_direction = "neutral"  # 上一次的趋势方向
        
        # 自动更新状态
        self.last_auto_update_time = 0
        
        # 价格穿越检测状态
        self.last_price = None  # 用于检测价格穿越
        self.price_crossed_up = False  # 价格上穿网格标志
        self.price_crossed_down = False  # 价格下穿网格标志
        
        # 异步边界调整状态
        self._pending_boundary_adjustment = False  # 是否有待处理的边界调整
        self._boundary_adjustment_in_progress = False  # 边界调整是否正在进行
        
        self.initialize_rate_sources()

    def initialize_rate_sources(self):
        self.market_data_provider.initialize_rate_sources([ConnectorPair(connector_name=self.config.connector_name,
                                                                         trading_pair=self.config.trading_pair)])
    
    def auto_detect_price_precision(self, price: Decimal) -> int:
        """自动检测价格精度"""
        if price is None or price <= 0:
            return 4  # 默认精度
        
        price_str = str(price)
        
        # 如果价格大于等于1，根据价格大小确定精度
        if price >= 1:
            if price >= 1000:
                return 2  # 高价格，如BTC: 50000.12
            elif price >= 100:
                return 3  # 中等价格，如ETH: 2500.123
            else:
                return 4  # 低价格，如BNB: 300.1234
        else:
            # 价格小于1，根据小数点后有效数字确定精度
            if '.' in price_str:
                decimal_part = price_str.split('.')[1]
                # 找到第一个非零数字的位置
                first_nonzero = 0
                for i, digit in enumerate(decimal_part):
                    if digit != '0':
                        first_nonzero = i
                        break
                
                # 对于小数价格，保留足够的有效数字
                if first_nonzero >= 6:  # 如PEPE: 0.000001234
                    return min(first_nonzero + 6, 12)  # 最多12位精度
                elif first_nonzero >= 3:  # 如DOGE: 0.001234
                    return first_nonzero + 4
                else:  # 如USDT: 0.9998
                    return 4
            else:
                return 4
    
    def get_price_precision(self) -> int:
        """获取当前使用的价格精度"""
        if self.price_precision is None:
            # 如果还没有检测过，使用默认精度
            return 4
        return self.price_precision

    def format_price(self, price: Decimal) -> Decimal:
        """根据自动检测的精度格式化价格"""
        if price is None:
            return None
        
        # 如果还没有检测过价格精度，先进行检测
        if self.price_precision is None:
            self.price_precision = self.auto_detect_price_precision(price)
        
        # 使用quantize方法将价格格式化为指定精度
        precision = Decimal('0.1') ** self.price_precision
        return price.quantize(precision)

    def active_executors(self) -> List[ExecutorInfo]:
        return [
            executor for executor in self.executors_info
            if executor.is_active
        ]

    def is_inside_bounds(self, price: Decimal) -> bool:
        if self.current_start_price is None or self.current_end_price is None:
            return False
        return self.current_start_price <= price <= self.current_end_price

    def update_price_history(self, current_price: Decimal):
        """更新价格历史记录"""
        current_time = time.time()
        self.price_history.append((current_time, current_price))
        
        # 清理过期数据
        cutoff_time = current_time - self.config.peak_detection_period
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff_time]

    def detect_peaks_and_valleys(self) -> tuple:
        """检测峰值和谷值"""
        if len(self.price_history) < 3:
            return None, None
            
        prices = [p for _, p in self.price_history]
        max_price = max(prices)
        min_price = min(prices)
        
        # 更新峰值和谷值
        if max_price > self.last_peak_price:
            self.last_peak_price = max_price
            
        if min_price < self.last_valley_price:
            self.last_valley_price = min_price
            
        return self.last_peak_price, self.last_valley_price

    def detect_price_crossing(self, current_price: Decimal):
        """
        检测价格是否穿越网格边界
        """
        if self.last_price is None or self.current_start_price is None or self.current_end_price is None:
            self.last_price = current_price
            return
            
        # 检测价格上穿网格上边界
        if (self.last_price <= self.current_end_price and 
            current_price > self.current_end_price):
            self.price_crossed_up = True
            self.price_crossed_down = False
            self.current_side = TradeType.BUY  # 价格上穿，开多
            
        # 检测价格下穿网格下边界
        elif (self.last_price >= self.current_start_price and 
              current_price < self.current_start_price):
            self.price_crossed_down = True
            self.price_crossed_up = False
            self.current_side = TradeType.SELL  # 价格下穿，开空
            
        # 价格在网格内部，重置穿越标志
        elif (self.current_start_price <= current_price <= self.current_end_price):
            self.price_crossed_up = False
            self.price_crossed_down = False
            self.current_side = None  # 价格在网格内，不操作
            
        self.last_price = current_price
        
    def update_trend_direction(self, current_price: Decimal):
        """更新趋势方向"""
        self.trend_prices.append(current_price)
        
        # 保持固定长度的价格序列
        if len(self.trend_prices) > self.config.trend_lookback_periods:
            self.trend_prices.pop(0)
            
        if len(self.trend_prices) >= self.config.trend_lookback_periods:
            # 计算趋势
            first_price = self.trend_prices[0]
            last_price = self.trend_prices[-1]
            price_change_pct = (last_price - first_price) / first_price
            
            if price_change_pct > self.config.trend_threshold:
                self.trend_direction = "up"
                self.current_side = TradeType.BUY  # 上涨趋势开多
            elif price_change_pct < -self.config.trend_threshold:
                self.trend_direction = "down"
                self.current_side = TradeType.SELL  # 下跌趋势开空
            else:
                self.trend_direction = "neutral"
                self.current_side = None  # 中性趋势不操作

    def should_auto_update(self) -> bool:
        """检查是否应该进行自动更新"""
        current_time = time.time()
        
        # 使用adjustment_interval作为统一的调整间隔
        if current_time - self.last_auto_update_time >= self.config.adjustment_interval:
            return True
            
        return False
    
    async def adjust_grid_boundaries(self, current_price: Decimal) -> bool:
        """根据峰值调整网格边界（安全版本）
        
        在调整网格边界前，会先确保：
        1. 停止所有活跃执行器
        2. 取消所有订单
        3. 等待持仓平仓
        
        Returns:
            bool: 调整是否成功
        """
        current_time = time.time()
        
        # 检查是否到了调整时间
        if current_time - self.last_adjustment_time < self.config.adjustment_interval:
            return False
            
        peak_price, valley_price = self.detect_peaks_and_valleys()
        
        if peak_price is None or valley_price is None:
            return False
        
        # 检查是否可以安全调整网格
        if not self.is_safe_to_adjust_grid():
            self.logger().info("Grid adjustment conditions not met, attempting to prepare for adjustment")
            
            # 尝试安全停止所有执行器
            success = await self.safe_stop_all_executors()
            if not success:
                self.logger().warning("Failed to safely stop executors, skipping grid adjustment")
                return False
            
            # 等待持仓平仓
            position_closed = await self.wait_for_position_close()
            if not position_closed:
                self.logger().warning("Position not fully closed, skipping grid adjustment")
                return False
            
            # 再次检查是否安全
            if not self.is_safe_to_adjust_grid():
                self.logger().warning("Still not safe to adjust grid after cleanup")
                return False
            
        self.logger().info("Safe to adjust grid boundaries - no active executors, orders, or positions")
            
        # 计算新的网格边界
        price_range = peak_price - valley_price
        grid_width = price_range * self.config.grid_width_percentage
        
        # 保存旧边界用于日志
        old_start = self.current_start_price
        old_end = self.current_end_price
        
        # 根据趋势方向调整边界
        if self.trend_direction == "up":
            # 上涨趋势：以当前价格为中心，向上偏移
            self.current_start_price = self.format_price(current_price - grid_width * Decimal("0.3"))
            self.current_end_price = self.format_price(current_price + grid_width * Decimal("0.7"))
        elif self.trend_direction == "down":
            # 下跌趋势：以当前价格为中心，向下偏移
            self.current_start_price = self.format_price(current_price - grid_width * Decimal("0.7"))
            self.current_end_price = self.format_price(current_price + grid_width * Decimal("0.3"))
        else:
            # 中性趋势：以当前价格为中心
            self.current_start_price = self.format_price(current_price - grid_width * Decimal("0.5"))
            self.current_end_price = self.format_price(current_price + grid_width * Decimal("0.5"))
            
        # 确保边界合理
        if self.current_start_price <= 0:
            self.current_start_price = self.format_price(current_price * Decimal("0.95"))
        
        # 记录边界调整
        precision = self.get_price_precision()
        self.logger().info(f"Grid boundaries adjusted safely:")
        self.logger().info(f"  Old: [{old_start:.{precision}f}, {old_end:.{precision}f}]")
        self.logger().info(f"  New: [{self.current_start_price:.{precision}f}, {self.current_end_price:.{precision}f}]")
        self.logger().info(f"  Current price: {current_price:.{precision}f}, Trend: {self.trend_direction}")
            
        self.last_adjustment_time = current_time
        self.last_auto_update_time = current_time
        self.last_trend_direction = self.trend_direction
        return True

    def should_restart_grid(self, current_price: Decimal) -> bool:
        """判断是否需要重启网格"""
        # 如果价格超出当前边界太多，需要重启
        if current_price < self.current_start_price * Decimal("0.95") or \
           current_price > self.current_end_price * Decimal("1.05"):
            return True
            
        # 如果趋势发生重大变化，需要重启
        active_executors = self.active_executors()
        if active_executors:
            executor = active_executors[0]
            if hasattr(executor.executor_config, 'side'):
                if executor.executor_config.side != self.current_side:
                    return True
                    
        return False
    
    def is_safe_to_adjust_grid(self) -> bool:
        """检查是否可以安全地调整网格
        
        要求：
        1. 没有活跃的执行器
        2. 没有未完成的订单
        3. 没有持仓
        
        Returns:
            bool: 如果可以安全调整网格返回True，否则返回False
        """
        # 检查是否有活跃的执行器
        active_executors = self.active_executors()
        if active_executors:
            self.logger().info(f"Cannot adjust grid: {len(active_executors)} active executors found")
            return False
        
        # 检查是否有未完成的订单
        connector = self.connectors[self.config.connector_name]
        open_orders = connector.get_open_orders()
        if open_orders:
            self.logger().info(f"Cannot adjust grid: {len(open_orders)} open orders found")
            return False
        
        # 检查是否有持仓
        if hasattr(connector, 'get_position'):
            position = connector.get_position(self.config.trading_pair)
            if position and abs(position.amount) > Decimal("0"):
                self.logger().info(f"Cannot adjust grid: Position exists with amount {position.amount}")
                return False
        
        return True
    
    async def safe_stop_all_executors(self) -> bool:
        """安全地停止所有执行器并等待完成
        
        Returns:
            bool: 如果成功停止所有执行器返回True
        """
        active_executors = self.active_executors()
        if not active_executors:
            return True
        
        self.logger().info(f"Stopping {len(active_executors)} active executors for grid adjustment")
        
        # 停止所有执行器
        stop_actions = []
        for executor in active_executors:
            # 使用early_stop方法立即停止执行器并取消所有订单
            executor.early_stop(keep_position=False)
            stop_actions.append(executor)
        
        # 等待执行器完全停止
        max_wait_time = 30  # 最大等待30秒
        wait_interval = 1   # 每秒检查一次
        waited_time = 0
        
        while waited_time < max_wait_time:
            remaining_executors = self.active_executors()
            if not remaining_executors:
                self.logger().info("All executors stopped successfully")
                return True
            
            self.logger().info(f"Waiting for {len(remaining_executors)} executors to stop... ({waited_time}s/{max_wait_time}s)")
            await asyncio.sleep(wait_interval)
            waited_time += wait_interval
        
        # 如果超时，强制取消所有订单
        self.logger().warning("Timeout waiting for executors to stop, forcing order cancellation")
        await self.force_cancel_all_orders()
        return False
    
    async def force_cancel_all_orders(self):
        """强制取消所有订单"""
        connector = self.connectors[self.config.connector_name]
        open_orders = connector.get_open_orders()
        
        if open_orders:
            self.logger().info(f"Force canceling {len(open_orders)} open orders")
            for order in open_orders:
                try:
                    connector.cancel(self.config.trading_pair, order.client_order_id)
                except Exception as e:
                    self.logger().error(f"Failed to cancel order {order.client_order_id}: {e}")
            
            # 等待订单取消完成
            await asyncio.sleep(2)
    
    async def wait_for_position_close(self, max_wait_time: int = 60) -> bool:
        """等待持仓完全平仓
        
        Args:
            max_wait_time: 最大等待时间（秒）
            
        Returns:
            bool: 如果持仓成功平仓返回True
        """
        connector = self.connectors[self.config.connector_name]
        wait_interval = 2
        waited_time = 0
        
        while waited_time < max_wait_time:
            if hasattr(connector, 'get_position'):
                position = connector.get_position(self.config.trading_pair)
                if not position or abs(position.amount) == Decimal("0"):
                    self.logger().info("Position successfully closed")
                    return True
                
                self.logger().info(f"Waiting for position to close... Current amount: {position.amount} ({waited_time}s/{max_wait_time}s)")
            else:
                # 如果不支持持仓查询，假设没有持仓
                return True
            
            await asyncio.sleep(wait_interval)
            waited_time += wait_interval
        
        self.logger().warning("Timeout waiting for position to close")
        return False

    def initialize_grid_boundaries(self, current_price: Decimal):
        """基于当前市场价格初始化网格边界"""
        if not self.is_initialized:
            grid_width = current_price * self.config.grid_width_percentage
            self.current_start_price = self.format_price(current_price - grid_width * Decimal("0.5"))
            self.current_end_price = self.format_price(current_price + grid_width * Decimal("0.5"))
            
            # 确保边界合理
            if self.current_start_price <= 0:
                self.current_start_price = self.format_price(current_price * Decimal("0.95"))
                
            self.is_initialized = True

    def determine_executor_actions(self) -> List[ExecutorAction]:
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
            
        # 初始化网格边界(仅在首次运行时)
        self.initialize_grid_boundaries(mid_price)
            
        # 更新价格历史
        self.update_price_history(mid_price)
        
        # 检测价格穿越网格边界
        self.detect_price_crossing(mid_price)
        
        # 检查是否需要自动更新
        should_update = self.should_auto_update()
        
        active_executors = self.active_executors()
        
        # 如果需要调整边界，但有活跃执行器，先停止它们
        if should_update and active_executors:
            self.logger().info("Grid boundary adjustment needed, stopping active executors first")
            return [StopExecutorAction(controller_id=self.config.id, executor_id=executor.id) 
                   for executor in active_executors]
        
        # 如果需要调整边界且没有活跃执行器，标记需要异步调整
        if should_update and not active_executors:
            # 设置标志，表示需要在下一个周期进行边界调整
            # 这里我们不直接调用异步方法，而是让后台任务处理
            self._pending_boundary_adjustment = True
            self.logger().info("Boundary adjustment scheduled for next cycle")
        
        # 如果需要重启网格，停止现有执行器
        if active_executors and self.should_restart_grid(mid_price):
            self.logger().info("Grid restart needed, stopping active executors")
            return [StopExecutorAction(controller_id=self.config.id, executor_id=executor.id) 
                   for executor in active_executors]
        
        # 如果没有活跃执行器且检测到价格穿越，创建新的网格执行器
        if (len(active_executors) == 0 and 
            (self.price_crossed_up or self.price_crossed_down) and 
            self.current_side is not None):
            
            # 确保在创建新执行器前是安全的
            if not self.is_safe_to_adjust_grid():
                self.logger().warning("Not safe to create new executor - waiting for cleanup")
                return []
            
            # 根据穿越方向调整网格边界
            if self.price_crossed_up:
                # 价格上穿，创建上方网格，开多
                grid_start = self.current_end_price
                grid_end = grid_start * (1 + self.config.grid_width_percentage)
                self.logger().info(f"Price crossed UP, creating LONG grid: [{self.format_price(grid_start)}, {self.format_price(grid_end)}]")
            else:  # self.price_crossed_down
                # 价格下穿，创建下方网格，开空
                grid_end = self.current_start_price
                grid_start = grid_end * (1 - self.config.grid_width_percentage)
                self.logger().info(f"Price crossed DOWN, creating SHORT grid: [{self.format_price(grid_start)}, {self.format_price(grid_end)}]")
            
            return [CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=GridExecutorConfig(
                    timestamp=self.market_data_provider.time(),
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.trading_pair,
                    start_price=self.format_price(grid_start),
                    end_price=self.format_price(grid_end),
                    leverage=self.config.leverage,
                    side=self.current_side,  # 根据穿越方向确定交易方向
                    total_amount_quote=self.config.total_amount_quote,

                    min_order_amount_quote=self.config.min_order_amount_quote,
                    max_open_orders=self.config.max_open_orders,
                    max_orders_per_batch=self.config.max_orders_per_batch,
                    order_frequency=self.config.order_frequency,
                    activation_bounds=self.config.activation_bounds,
                    triple_barrier_config=self.config.triple_barrier_config,
                    level_id=None,
                    keep_position=self.config.keep_position,
                ))]
        return []

    async def update_processed_data(self):
        """处理异步任务，包括边界调整"""
        # 处理待处理的边界调整
        if self._pending_boundary_adjustment and not self._boundary_adjustment_in_progress:
            await self._handle_boundary_adjustment()
    
    async def _handle_boundary_adjustment(self):
        """处理边界调整的异步任务"""
        if self._boundary_adjustment_in_progress:
            return
        
        self._boundary_adjustment_in_progress = True
        self._pending_boundary_adjustment = False
        
        try:
            mid_price = self.market_data_provider.get_price_by_type(
                self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
            
            self.logger().info("Starting asynchronous boundary adjustment")
            success = await self.adjust_grid_boundaries(mid_price)
            
            if success:
                self.logger().info("Boundary adjustment completed successfully")
            else:
                self.logger().warning("Boundary adjustment failed or was skipped")
                
        except Exception as e:
            self.logger().error(f"Error during boundary adjustment: {e}")
        finally:
            self._boundary_adjustment_in_progress = False

    def to_format_status(self) -> List[str]:
        status = []
        mid_price = self.market_data_provider.get_price_by_type(
            self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
            
        # 定义标准框宽度
        box_width = 120
        
        # 动态网格配置框
        status.append("┌" + "─" * box_width + "┐")
        
        # 第一行：动态网格配置
        left_section = "Dynamic Grid Configuration:"
        padding = box_width - len(left_section) - 4
        config_line1 = f"│ {left_section}{' ' * padding} │"
        status.append(config_line1)
        
        # 第二行：当前边界和趋势
        precision = self.get_price_precision()
        start_str = f"{self.current_start_price:.{precision}f}" if self.current_start_price else "Not Set"
        end_str = f"{self.current_end_price:.{precision}f}" if self.current_end_price else "Not Set"
        side_str = str(self.current_side) if self.current_side else "No Action"
        config_line2 = f"│ Start: {start_str} │ End: {end_str} │ Side: {side_str} │ Trend: {self.trend_direction} │ Mid: {mid_price:.{precision}f} │"
        padding = box_width - len(config_line2) + 1
        config_line2 += " " * padding + "│"
        status.append(config_line2)
        
        # 第三行：峰值信息和精度信息
        config_line3 = f"│ Peak: {self.last_peak_price:.{precision}f} │ Valley: {self.last_valley_price:.{precision}f} │ Precision: {precision} │ Inside bounds: {1 if self.is_inside_bounds(mid_price) else 0} │"
        padding = box_width - len(config_line3) + 1
        config_line3 += " " * padding + "│"
        status.append(config_line3)
        
        # 第四行：价格穿越状态信息
        cross_status = "None"
        if self.price_crossed_up:
            cross_status = "UP (Ready to LONG)"
        elif self.price_crossed_down:
            cross_status = "DOWN (Ready to SHORT)"
        config_line4 = f"│ Price Crossing: {cross_status} │ Strategy: Price crossing based grid trading │"
        padding = box_width - len(config_line4) + 1
        config_line4 += " " * padding + "│"
        status.append(config_line4)
        
        # 第五行：调整信息
        last_adj_time = time.time() - self.last_adjustment_time if self.last_adjustment_time > 0 else 0
        config_line5 = f"│ Last Adjustment: {last_adj_time:.0f}s ago │ Price History: {len(self.price_history)} points │"
        padding = box_width - len(config_line5) + 1
        config_line5 += " " * padding + "│"
        status.append(config_line5)
        
        status.append("└" + "─" * box_width + "┘")
        
        # 显示活跃执行器状态
        for level in self.active_executors():
            total_width = box_width
            status_header = f"Dynamic Grid Status: {level.id} (RUNNING)"
            status_line = f"┌ {status_header}" + "─" * (total_width - len(status_header) - 2) + "┐"
            status.append(status_line)
            
            # 执行器详细信息
            if hasattr(level, 'custom_info') and level.custom_info:
                # 三列布局
                col_width = box_width // 3
                
                # 列标题
                header_line = "│ Level Distribution" + " " * (col_width - 20) + "│"
                header_line += " Order Statistics" + " " * (col_width - 18) + "│"
                header_line += " Performance Metrics" + " " * (col_width - 21) + "│"
                status.append(header_line)
                
                # 数据行
                level_dist_data = [
                    f"NOT_ACTIVE: {len(level.custom_info.get('levels_by_state', {}).get('NOT_ACTIVE', []))}",
                    f"OPEN_ORDER_PLACED: {len(level.custom_info.get('levels_by_state', {}).get('OPEN_ORDER_PLACED', []))}",
                    f"COMPLETE: {len(level.custom_info.get('levels_by_state', {}).get('COMPLETE', []))}"
                ]
                
                order_stats_data = [
                    f"Filled: {len(level.custom_info.get('filled_orders', []))}",
                    f"Failed: {len(level.custom_info.get('failed_orders', []))}",
                    f"Canceled: {len(level.custom_info.get('canceled_orders', []))}"
                ]
                
                perf_metrics_data = [
                    f"R. PnL: {level.custom_info.get('realized_pnl_quote', 0):.4f}",
                    f"Position: {level.custom_info.get('position_size_quote', 0):.4f}",
                    f"P. PnL: {level.custom_info.get('position_pnl_quote', 0):.4f}"
                ]
                
                max_rows = max(len(level_dist_data), len(order_stats_data), len(perf_metrics_data))
                for i in range(max_rows):
                    col1 = level_dist_data[i] if i < len(level_dist_data) else ""
                    col2 = order_stats_data[i] if i < len(order_stats_data) else ""
                    col3 = perf_metrics_data[i] if i < len(perf_metrics_data) else ""
                    
                    row = "│ " + col1
                    row += " " * (col_width - len(col1) - 2)
                    row += "│ " + col2
                    row += " " * (col_width - len(col2) - 2)
                    row += "│ " + col3
                    row += " " * (col_width - len(col3) - 2)
                    row += "│"
                    status.append(row)
            
            status.append("└" + "─" * total_width + "┘")
            
        return status