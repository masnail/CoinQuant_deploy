from typing import List

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

class TrendFollowingConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "trend_following_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    # MA
    ma1: int = Field(
        default=3,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    ma2: int = Field(
        default=5,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    ma3: int = Field(
        default=7,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})

    # MACD
    macd_fast: int = Field(
        default=12,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    macd_slow: int = Field(
        default=26,
        json_schema_extra={"prompt": "Enter the MACD slow period: ", "prompt_on_new": True})
    macd_signal: int = Field(
        default=9,
        json_schema_extra={"prompt": "Enter the MACD signal period: ", "prompt_on_new": True})

    # RSI
    rsi_period: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the RSI period: ", "prompt_on_new": True})
    rsi_upper: int = Field(
        default=70,
        json_schema_extra={"prompt": "Enter the RSI upper threshold: ", "prompt_on_new": True})
    rsi_lower: int = Field(
        default=30,
        json_schema_extra={"prompt": "Enter the RSI lower threshold: ", "prompt_on_new": True})

    # EMA
    ema1: int = Field(
        default=50,
        json_schema_extra={"prompt": "Enter the EMA1 period: ", "prompt_on_new": True})
    ema2: int = Field(
        default=100,
        json_schema_extra={"prompt": "Enter the EMA2 period: ", "prompt_on_new": True})
    ema3: int = Field(
        default=200,
        json_schema_extra={"prompt": "Enter the EMA3 period: ", "prompt_on_new": True})

    # ADX
    adx_period: int = Field(
        default=14,
        json_schema_extra={"prompt": "Enter the ADX period: ", "prompt_on_new": True})
    adx_threshold: int = Field(
        default=70,
        json_schema_extra={"prompt": "Enter the ADX threshold: ", "prompt_on_new": True})

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class TrendFollowingController(DirectionalTradingControllerBase):

    def __init__(self, config: TrendFollowingConfig, *args, **kwargs):
        self.config = config
        self.max_records = 100
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        
        # self.logger().info("trend_following_v1 update_processed_data")

        # MA
        df.ta.sma(length=self.config.ma1, append=True)
        df.ta.sma(length=self.config.ma2, append=True)
        df.ta.sma(length=self.config.ma3, append=True)
        ma1 = df[f"SMA_{self.config.ma1}"]
        ma2 = df[f"SMA_{self.config.ma2}"]
        ma3 = df[f"SMA_{self.config.ma3}"]
        
        # MACD
        df.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal, append=True)

        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macd = df[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]

        # RSI
        df.ta.rsi(length=self.config.rsi_period, append=True)
        rsi = df[f"RSI_{self.config.rsi_period}"]

        # EMA
        df.ta.ema(length=self.config.ema1, append=True)
        df.ta.ema(length=self.config.ema2, append=True)
        df.ta.ema(length=self.config.ema3, append=True)
        ema1 = df[f"EMA_{self.config.ema1}"]
        ema2 = df[f"EMA_{self.config.ema2}"]
        ema3 = df[f"EMA_{self.config.ema3}"]

        # ADX
        df.ta.adx(length=self.config.adx_period, append=True)
        adx = df[f"ADX_{self.config.adx_period}"]

        # Generate signal
        long_condition = ((ma1 > ma2) & (ma2 > ma3)) & (macdh > 0) & (macd < 0) & (rsi < self.config.rsi_lower) & (adx > self.config.adx_threshold) & ((ema1 > ema2) & (ema2 > ema3))
        short_condition = ((ma3 > ma2) & (ma2 > ma1)) & (macdh < 0) & (macd > 0) & (rsi > self.config.rsi_upper) & (adx > self.config.adx_threshold) & ((ema3 > ema2) & (ema2 > ema1))

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
