"""
盘中分钟价小数位与前复权基准价对比。

用于在本地回测与聚宽环境输出同样的检查结果，便于对齐差异。
"""
from jqdata import *  # noqa: F401,F403

import numpy as np
import pandas as pd


STOCK_CODE = "300640.XSHE"
FUND_CODE = "511880.XSHG"
TARGET_DATE = "2024-04-01"


def _maybe_set_data_provider():
    setter = globals().get("set_data_provider")
    if callable(setter):
        try:
            setter("jqdata")
        except Exception:
            pass


def _init_results():
    g.results = {
        "minute_checks": [],
        "pre_close": None,
        "errors": [],
    }
    g._minute_checked = False
    g._pre_checked = False


def _record_error(message):
    g.results["errors"].append(message)
    log.error(message)


def _infer_decimals(code):
    info = None
    try:
        info = get_security_info(code)
    except Exception:
        info = None

    for attr in ("price_decimals", "tick_decimals"):
        if info is not None and hasattr(info, attr):
            try:
                val = int(getattr(info, attr))
                if val >= 0:
                    return val
            except Exception:
                pass

    if isinstance(info, dict):
        for key in ("price_decimals", "tick_decimals"):
            val = info.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                return int(val)
        category = info.get("category") or info.get("type")
    else:
        category = getattr(info, "category", None) or getattr(info, "type", None)

    if category and str(category).lower() in ("fund", "money_market_fund", "etf"):
        return 3
    if code.startswith("511"):
        return 3
    return 2


def _extract_close(df):
    if df is None:
        return None
    try:
        if df.empty:
            return None
    except Exception:
        pass

    try:
        value = df.iloc[-1]["close"]
    except Exception:
        try:
            value = df.iloc[-1][0]
        except Exception:
            return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def initialize(context):
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    set_benchmark("000300.XSHG")

    _maybe_set_data_provider()
    _init_results()

    g.targets = [STOCK_CODE, FUND_CODE]
    run_daily(check_minute_close, time="14:25")
    run_daily(check_pre_close, time="close")


def check_minute_close(context):
    if getattr(g, "_minute_checked", False):
        return
    g._minute_checked = True
    from datetime import timedelta
    start_time = context.current_dt - timedelta(minutes=2)
    for code in g.targets:
        try:
            df = get_price(
                security=code,
                end_date=context.current_dt,
                frequency="minute",
                fields=["close"],
                count=1,
                fq="pre",
            )
            raw_close = _extract_close(df)
            if raw_close is None:
                _record_error(f"分钟行情为空: {code}")
                continue

            decimals = _infer_decimals(code)
            rounded = float(np.round(raw_close, decimals))
            diff = abs(raw_close - rounded)
            payload = {
                "code": code,
                "raw_close": raw_close,
                "rounded_close": rounded,
                "decimals": decimals,
                "diff": diff,
                "time": str(context.current_dt),
            }
            g.results["minute_checks"].append(payload)
            log.info(
                "[精度对比][分钟] %s raw=%.6f rounded=%.6f decimals=%s diff=%.10f",
                code,
                raw_close,
                rounded,
                decimals,
                diff,
            )
        except Exception as exc:
            _record_error(f"分钟行情失败 {code}: {exc}")


def check_pre_close(context):
    if getattr(g, "_pre_checked", False):
        return
    g._pre_checked = True

    try:
        df = get_price(
            security=STOCK_CODE,
            end_date=TARGET_DATE,
            frequency="daily",
            fields=["close"],
            count=1,
            fq="pre",
        )
        raw_close = _extract_close(df)
        if raw_close is None:
            _record_error(f"前复权行情为空: {STOCK_CODE}")
            return

        decimals = _infer_decimals(STOCK_CODE)
        rounded = float(np.round(raw_close, decimals))
        g.results["pre_close"] = {
            "code": STOCK_CODE,
            "raw_close": raw_close,
            "rounded_close": rounded,
            "decimals": decimals,
            "target_date": TARGET_DATE,
            "context_date": str(context.current_dt.date()),
        }
        log.info(
            "[精度对比][前复权] %s date=%s raw=%.6f rounded=%.6f decimals=%s ref=%s",
            STOCK_CODE,
            TARGET_DATE,
            raw_close,
            rounded,
            decimals,
            context.current_dt.date(),
        )
    except Exception as exc:
        _record_error(f"前复权行情失败 {STOCK_CODE}: {exc}")
