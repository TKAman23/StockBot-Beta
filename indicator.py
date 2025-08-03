import numpy as np
import pandas as pd
from tech_indicators import (
    sma,
    ema,
    macd,
    rsi,
    stochastic_rsi,
    roc,
    mom,
    cmf,
    obv,
    mfi,
    bollinger_bands,
    atr,
    donchian_channels,
    trix,
    fibonacci_retracements,
    williams_fractals,
    price_channel
)

def add_indicators(
    in_data: pd.DataFrame,
    smap: set = {10, 20, 50, 200},
    emap: set = {9, 12, 26, 50},
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_window: int = 14,
    srsi_window: int = 14,
    srsi_k: int = 3,
    srsi_d: int = 3,
    roc_window: int = 12,
    mom_window: int = 10,
    cmf_window: int = 20,
    mfi_window: int = 14,
    bollinger_window: int = 20,
    bollinger_num_std: int = 2,
    inc_obv: bool = True,
    atr_window: int = 14,
    donchian_window: int = 20,
    trix_window: int = 15,
    inc_fib: bool = True,
    pc_window: int = 20,
    inc_fractals: bool = True,
) -> pd.DataFrame:
    data = in_data.copy()

    # Add SMA
    if smap:
        for window in smap:
            data[f"sma {window}"] = sma(series=data["close"], window=window)

    # Add EMA
    if emap:
        for window in emap:
            data[f"ema {window}"] = ema(series=data["close"], window=window)

    # Add MACD
    if macd_fast and macd_slow:
        data["macd line"], data["macd signal"] = macd(series=data["close"], fast=macd_fast, slow=macd_slow)

    # Add RSI
    if rsi_window:
        data["rsi"] = rsi(series=data["close"], window=rsi_window)

    # Add Stochastic RSI
    if srsi_window and srsi_k and srsi_d:
        data["stochastic rsi k"], data["stochastic rsi d"] = stochastic_rsi(series=data["close"], window=srsi_window, smooth_k=srsi_k, smooth_d=srsi_d)

    # Add ROC
    if roc_window:
        data["roc"] = roc(series=data["close"], window=roc_window)

    # Add Momentum
    if mom_window:
        data["momentum"] = mom(series=data["close"], window=mom_window)

    # Add CMF
    if cmf_window:
        data["cmf"] = cmf(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=cmf_window)

    # Add OBV
    if inc_obv:
        data["obv"] = obv(close=data["close"], volume=data["volume"])

    # Add MFI
    if mfi_window:
        data["mfi"] = mfi(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=mfi_window)

    # Add Bollinger Bands
    if bollinger_window and bollinger_num_std:
        data["bollinger upper"], data["bollinger lower"] = bollinger_bands(series=data["close"], window=bollinger_window, num_std=bollinger_num_std)

    # Add ATR
    if atr_window:
        data["atr"] = atr(high=data["high"], low=data["low"], close=data["close"], window=atr_window)

    # Add Donchian Channels
    if donchian_window:
        data["donchian upper"], data["donchian lower"] = donchian_channels(series=data["close"], window=donchian_window)

    # Add TRIX
    if trix_window:
        data["trix"] = trix(series=data["close"], window=trix_window)

    # Add Williams Fractals
    fractals_up, fractals_down = williams_fractals(high=data["high"], low=data["low"])
    if inc_fractals:
        data["williams up"] = fractals_up.astype('int8')
        data["williams down"] = fractals_down.astype('int8')

    # Add Fib
    if inc_fib:
        data = pd.concat([
                data,
                fibonacci_retracements(
                    high=data["high"],
                    low=data["low"],
                    fractals_up=fractals_up,
                    fractals_down=fractals_down)],
            axis=1)

    if pc_window:
        data = pd.concat(
            [data, price_channel(high=data["high"], low=data["low"], window=pc_window)],
            axis=1
        )

    return data