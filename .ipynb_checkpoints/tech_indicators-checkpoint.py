import pandas as pd
import numpy as np
from types import GeneratorType

def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA)
    Parameters:
    - series: price series (e.g., close prices)
    - window: number of periods
    Returns:
    - SMA series
    """
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    Parameters:
    - series: price series
    - span: span for the exponential window (smoothing)
    Returns:
    - EMA series
    """
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> (pd.Series, pd.Series):
    """
    Moving Average Convergence Divergence (MACD)
    Parameters:
    - series: price series
    - fast: period for fast EMA
    - slow: period for slow EMA
    - signal: period for signal line EMA
    Returns:
    - macd_line: difference between fast and slow EMA
    - signal_line: EMA of macd_line
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    Parameters:
    - series: price series
    - window: lookback period
    Returns:
    - RSI series (0-100)
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def stochastic_rsi(series: pd.Series, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> (pd.Series, pd.Series):
    """
    Stochastic RSI
    Parameters:
    - series: price series
    - window: RSI lookback
    - smooth_k: %K smoothing
    - smooth_d: %D smoothing
    Returns:
    - stoch_k: stochastic RSI %K
    - stoch_d: stochastic RSI %D (signal)
    """
    rsi_series = rsi(series, window)
    min_rsi = rsi_series.rolling(window).min()
    max_rsi = rsi_series.rolling(window).max()
    stoch_k = ((rsi_series - min_rsi) / (max_rsi - min_rsi)).fillna(0)
    stoch_k = stoch_k.rolling(smooth_k).mean()
    stoch_d = stoch_k.rolling(smooth_d).mean()
    return stoch_k, stoch_d


def roc(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Rate of Change (ROC)
    Parameters:
    - series: price series
    - period: lookback period
    Returns:
    - ROC as percentage
    """
    return series.pct_change(periods=window) * 100


def mom(series: pd.Series, window: int = 10) -> pd.Series:
    """
    Momentum (MOM)
    Parameters:
    - series: price series
    - period: lookback period
    Returns:
    - momentum values
    """
    return series.diff(window)


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF)
    Parameters:
    - high, low, close, volume: price & volume series
    - window: lookback period
    Returns:
    - CMF series
    """
    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    return mfv.rolling(window).sum() / volume.rolling(window).sum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)
    Parameters:
    - close: price series
    - volume: volume series
    Returns:
    - OBV series
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI)
    Parameters:
    - high, low, close: price series
    - volume: volume series
    - window: lookback period
    Returns:
    - MFI series (0-100)
    """
    tp = (high + low + close) / 3
    mf = tp * volume
    positive_mf = mf.where(tp > tp.shift(1), 0).rolling(window).sum()
    negative_mf = mf.where(tp < tp.shift(1), 0).rolling(window).sum()
    mfr = positive_mf / negative_mf
    return 100 - (100 / (1 + mfr))


def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> (pd.Series, pd.Series):
    """
    Bollinger Bands
    Parameters:
    - series: price series
    - window: moving average window
    - num_std: number of standard deviations
    Returns:
    - upper_band, lower_band
    """
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    Parameters:
    - high, low, close: price series
    - window: lookback period
    Returns:
    - ATR series
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def donchian_channels(series: pd.Series, window: int = 20) -> (pd.Series, pd.Series):
    """
    Donchian Channels
    Parameters:
    - series: price series (typically high or low series separately)
    - window: lookback period
    Returns:
    - upper_channel, lower_channel
    """
    upper = series.rolling(window).max()
    lower = series.rolling(window).min()
    return upper, lower


def trix(series: pd.Series, window: int = 15) -> pd.Series:
    """
    TRIX (Triple Exponential Average)
    Parameters:
    - series: price series
    - window: EMA window
    Returns:
    - TRIX series
    """
    e1 = ema(series, window)
    e2 = ema(e1, window)
    e3 = ema(e2, window)
    return e3.pct_change() * 100


def williams_fractals(high: pd.Series, low: pd.Series) -> (pd.Series, pd.Series):
    """
    Calculate Williams Fractals (Up and Down) for a given OHLC DataFrame.
    Assumes DataFrame has columns: 'High' and 'Low'.

    Returns:
        DataFrame with additional columns: 'Fractal_Up' and 'Fractal_Down'
    """

    # Fractal Up: High[i] > High[i-1], High[i-2], High[i+1], High[i+2]
    fractal_up = (
        (high.shift(2) < high.shift(0)) &
        (high.shift(1) < high.shift(0)) &
        (high.shift(-1) < high.shift(0)) &
        (high.shift(-2) < high.shift(0))
    )

    # Fractal Down: Low[i] < Low[i-1], Low[i-2], Low[i+1], Low[i+2]
    fractal_down = (
        (low.shift(2) > low.shift(0)) &
        (low.shift(1) > low.shift(0)) &
        (low.shift(-1) > low.shift(0)) &
        (low.shift(-2) > low.shift(0))
    )

    return fractal_up, fractal_down


def fibonacci_retracements(high: pd.Series, low: pd.Series, fractals_up: pd.Series, fractals_down: pd.Series) -> pd.DataFrame:
    """
    Fibonacci Retracement Levels
    Parameters:
     - high: daily highs
     - low: daily lows
     - fractals_up: williams fractals up signal
     - fractals_down: williams fractals down signal
    Returns:
     - levels: fib level using most recent williams fractal signals
    """
    # Getting high and low values filled
    fractals_high = high.where(fractals_up).pipe(lambda s: _fib_fill_na(s))
    fractals_low = low.where(fractals_down).pipe(lambda s: _fib_fill_na(s))

    # Fixing generator type issue:
    if isinstance(high, GeneratorType):
        high = pd.Series(list(high))
    if isinstance(low, GeneratorType):
        low = pd.Series(list(low))
    
    return pd.concat([
        pd.DataFrame(
            _fib_down(fractals_high, fractals_low), 
            _fib_down(fractals_high, fractals_low),
        ),
        pd.DataFrame(
            _fib_up(fractals_high, fractals_low),
            _fib_down(fractals_high, fractals_low),
        ),
    ],
    axis=1)

def _fib_down(high: pd.Series, low: pd.Series) -> dict:
    """
    Fibonacci Retracement Levels
    Parameters:
    - high: swing high price
    - low: swing low price
    Returns:
    - levels: dict mapping Fibonacci ratio to price level
    """
    diff = high - low
    return {
        "fibonacci down 0.0" : high,
        "fibonacci down 0.236" : high - 0.236 * diff,
        "fibonacci down 0.382" : high - 0.382 * diff,
        "fibonacci down 0.5" : high - 0.5 * diff,
        "fibonacci down 0.618" : high - 0.618 * diff,
        "fibonacci down 0.786" : high - 0.786 * diff,
        "fibonacci down 1.0" : low
    }

def _fib_up(high: pd.Series, low: pd.Series) -> dict:
    """
    Fibonacci Retracement Levels
    Parameters:
    - high: swing high price
    - low: swing low price
    Returns:
    - levels: dict mapping Fibonacci ratio to price level
    """
    diff = high - low
    return {
        "fibonacci up 0.0" : low,
        "fibonacci up 0.236" : low + 0.236 * diff,
        "fibonacci up 0.382" : low + 0.382 * diff,
        "fibonacci up 0.5" : low + 0.5 * diff,
        "fibonacci up 0.618" : low + 0.618 * diff,
        "fibonacci up 0.786" : low + 0.786 * diff,
        "fibonacci up 1.0" : high
    }

def _fib_fill_na(series):
    filled = []
    last_value = np.nan
    for elem in series:
        if pd.isna(elem):
            filled.append(last_value)
        else:
            filled.append(elem)
            last_value = elem
    return pd.Series(filled)

# Causing generator type error
# def _fib_fill_na(elem, reset=False):
#     e = np.nan
#     while True:
#         if pd.isna(elem):
#             yield e
#             if reset:
#                 return
#         else:
#             e = yield elem


def price_channel(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """
    Price Channels
    Parameters:
    - high: high price series
    - low: low price series
    - window: lookback period
    Returns:
    - DataFrame with columns ['Upper', 'Lower', 'Mid']
    """
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    mid = (upper + lower) / 2
    return pd.DataFrame({'upper': upper, 'lower': lower, 'mid': mid})
