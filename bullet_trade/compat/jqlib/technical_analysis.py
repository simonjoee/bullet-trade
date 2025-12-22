"""
jqlib.technical_analysis 兼容模块

提供与聚宽 jqlib.technical_analysis 兼容的技术分析函数
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional

# 尝试导入 talib，如果不可用则使用 numpy 实现
try:
    import talib
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


def _sma(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """简单移动平均"""
    if _HAS_TALIB:
        return talib.SMA(close, timeperiod=timeperiod)
    else:
        result = np.full(len(close), np.nan)
        for i in range(timeperiod - 1, len(close)):
            result[i] = np.mean(close[i - timeperiod + 1:i + 1])
        return result


def _ema(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """指数移动平均"""
    if _HAS_TALIB:
        return talib.EMA(close, timeperiod=timeperiod)
    else:
        result = np.full(len(close), np.nan)
        alpha = 2.0 / (timeperiod + 1.0)
        result[timeperiod - 1] = np.mean(close[:timeperiod])
        for i in range(timeperiod, len(close)):
            result[i] = alpha * close[i] + (1 - alpha) * result[i - 1]
        return result


# 移动平均线
def MA(close: Union[np.ndarray, pd.Series], timeperiod: int = 5) -> np.ndarray:
    """
    简单移动平均线
    
    Args:
        close: 收盘价序列
        timeperiod: 周期，默认5
        
    Returns:
        移动平均线数组
    """
    if isinstance(close, pd.Series):
        close = close.values
    return _sma(close, timeperiod)


def EMA(close: Union[np.ndarray, pd.Series], timeperiod: int = 5) -> np.ndarray:
    """
    指数移动平均线
    
    Args:
        close: 收盘价序列
        timeperiod: 周期，默认5
        
    Returns:
        指数移动平均线数组
    """
    if isinstance(close, pd.Series):
        close = close.values
    return _ema(close, timeperiod)


def SMA(close: Union[np.ndarray, pd.Series], timeperiod: int = 5) -> np.ndarray:
    """简单移动平均线（别名）"""
    return MA(close, timeperiod)


# MACD
def MACD(close: Union[np.ndarray, pd.Series], 
         fastperiod: int = 12, 
         slowperiod: int = 26, 
         signalperiod: int = 9) -> tuple:
    """
    MACD指标
    
    Args:
        close: 收盘价序列
        fastperiod: 快线周期，默认12
        slowperiod: 慢线周期，默认26
        signalperiod: 信号线周期，默认9
        
    Returns:
        (DIF, DEA, MACD) 元组
    """
    if isinstance(close, pd.Series):
        close = close.values
    
    if _HAS_TALIB:
        dif, dea, macd = talib.MACD(close, fastperiod=fastperiod, 
                                    slowperiod=slowperiod, 
                                    signalperiod=signalperiod)
        return dif, dea, macd
    else:
        # 使用 EMA 实现
        ema_fast = _ema(close, fastperiod)
        ema_slow = _ema(close, slowperiod)
        dif = ema_fast - ema_slow
        
        # DEA 是 DIF 的 EMA
        dea = _ema(dif, signalperiod)
        
        # MACD = (DIF - DEA) * 2
        macd = (dif - dea) * 2
        
        return dif, dea, macd


# RSI
def RSI(close: Union[np.ndarray, pd.Series], timeperiod: int = 14) -> np.ndarray:
    """
    相对强弱指标
    
    Args:
        close: 收盘价序列
        timeperiod: 周期，默认14
        
    Returns:
        RSI数组
    """
    if isinstance(close, pd.Series):
        close = close.values
    
    if _HAS_TALIB:
        return talib.RSI(close, timeperiod=timeperiod)
    else:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        result = np.full(len(close), np.nan)
        for i in range(timeperiod, len(close)):
            avg_gain = np.mean(gain[i - timeperiod:i])
            avg_loss = np.mean(loss[i - timeperiod:i])
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        return result


# BOLL
def BOLL(close: Union[np.ndarray, pd.Series], 
         timeperiod: int = 20, 
         nbdevup: float = 2.0, 
         nbdevdn: float = 2.0) -> tuple:
    """
    布林带
    
    Args:
        close: 收盘价序列
        timeperiod: 周期，默认20
        nbdevup: 上轨标准差倍数，默认2.0
        nbdevdn: 下轨标准差倍数，默认2.0
        
    Returns:
        (上轨, 中轨, 下轨) 元组
    """
    if isinstance(close, pd.Series):
        close = close.values
    
    if _HAS_TALIB:
        upper, middle, lower = talib.BBANDS(close, timeperiod=timeperiod,
                                            nbdevup=nbdevup, nbdevdn=nbdevdn)
        return upper, middle, lower
    else:
        middle = _sma(close, timeperiod)
        std = np.full(len(close), np.nan)
        for i in range(timeperiod - 1, len(close)):
            std[i] = np.std(close[i - timeperiod + 1:i + 1])
        upper = middle + nbdevup * std
        lower = middle - nbdevdn * std
        return upper, middle, lower


# KDJ
def KDJ(high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        fastk_period: int = 9,
        slowk_period: int = 3,
        slowd_period: int = 3) -> tuple:
    """
    KDJ指标
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        fastk_period: 快速K周期，默认9
        slowk_period: 慢速K周期，默认3
        slowd_period: 慢速D周期，默认3
        
    Returns:
        (K, D, J) 元组
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    if _HAS_TALIB:
        k, d = talib.STOCH(high, low, close,
                          fastk_period=fastk_period,
                          slowk_period=slowk_period,
                          slowd_period=slowd_period)
        j = 3 * k - 2 * d
        return k, d, j
    else:
        # 简化实现
        k = np.full(len(close), np.nan)
        for i in range(fastk_period - 1, len(close)):
            highest = np.max(high[i - fastk_period + 1:i + 1])
            lowest = np.min(low[i - fastk_period + 1:i + 1])
            if highest != lowest:
                rsv = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv = 50
            k[i] = rsv
        
        # K 的移动平均得到 D
        d = _sma(k, slowk_period)
        # J = 3K - 2D
        j = 3 * k - 2 * d
        return k, d, j


# 导出所有函数
__all__ = [
    'MA', 'EMA', 'SMA',
    'MACD',
    'RSI',
    'BOLL',
    'KDJ',
]

