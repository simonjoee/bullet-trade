"""
订单系统

提供各种下单函数
"""

from dataclasses import dataclass
from typing import Optional, Union
from datetime import datetime
import uuid
import asyncio
import inspect

from .models import Order, OrderStatus, OrderStyle
from .globals import log
from .settings import get_settings
from .runtime import process_orders_now, get_current_engine


# 全局订单队列
_order_queue = []


@dataclass
class MarketOrderStyle:
    """市价单参数，可指定保护价或买卖价差。"""
    limit_price: Optional[float] = None
    buy_price_percent: Optional[float] = None
    sell_price_percent: Optional[float] = None


@dataclass
class LimitOrderStyle:
    """限价单参数：显式给出委托价格。"""
    price: float


def _generate_order_id() -> str:
    """生成唯一订单ID"""
    return str(uuid.uuid4())


def _trigger_order_processing(wait_timeout: Optional[float] = None) -> None:
    """
    触发订单处理：
    - 实盘：将 _process_orders 投递到事件循环，避免当前协程阻塞；
    - 回测/模拟：order_match_mode=immediate 时同步处理。
    """
    try:
        settings = get_settings()
        engine = get_current_engine()
        if getattr(engine, "is_live", False):
            loop = getattr(engine, "_loop", None)
            if loop and loop.is_running():
                wait_for_result = wait_timeout is None or wait_timeout > 0
                try:
                    try:
                        running_loop = asyncio.get_running_loop()
                    except RuntimeError:
                        running_loop = None

                    if wait_for_result and running_loop is None:
                        fut = asyncio.run_coroutine_threadsafe(
                            engine._process_orders(engine.context.current_dt),
                            loop,
                        )
                        fut.result(timeout=wait_timeout if wait_timeout and wait_timeout > 0 else None)
                    else:
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(engine._process_orders(engine.context.current_dt))
                        )
                except Exception as exc:
                    log.debug(f"投递实盘订单处理任务失败: {exc}")
                return
        if settings.options.get('order_match_mode') == 'immediate':
            process_orders_now()
    except Exception as e:
        log.warning(f"触发订单处理失败，保留到队列: {e}")


def order(
    security: str,
    amount: int,
    price: Optional[float] = None,
    style: Optional[Union[OrderStyle, MarketOrderStyle, LimitOrderStyle]] = None,
    wait_timeout: Optional[float] = None,
) -> Optional[Order]:
    """
    按股数下单
    
    Args:
        security: 标的代码
        amount: 股数，正数表示买入，负数表示卖出
        price: 委托价格，None表示市价单
        style: 下单方式或市价参数（策略覆写）
        
    Returns:
        Order对象，如果下单失败返回None
    """
    if isinstance(price, (MarketOrderStyle, LimitOrderStyle)):
        style = price
        price = None

    if amount == 0:
        log.warning(f"下单数量为0，忽略订单: {security}")
        return None
    
    if style is not None:
        resolved_style: object = style
    elif price is not None:
        # 未显式传入限价风格时，按市价单带保护价处理
        resolved_style = MarketOrderStyle(limit_price=price)
    else:
        resolved_style = MarketOrderStyle()

    order_obj = Order(
        order_id=_generate_order_id(),
        security=security,
        amount=abs(amount),
        price=price if price is not None else 0.0,
        status=OrderStatus.open,
        add_time=datetime.now(),
        is_buy=(amount > 0),
        style=resolved_style,
        wait_timeout=wait_timeout,
    )
    
    _order_queue.append(order_obj)
    log.debug(f"创建订单: {security}, 数量: {amount}, 价格: {price}")
    _trigger_order_processing(wait_timeout)
    
    return order_obj


def cancel_order(order_or_id: Union[Order, str]) -> bool:
    """
    撤单：优先取消本地队列订单，若已下到券商且有券商订单号则调用券商撤单。
    
    Args:
        order_or_id: Order 对象或订单 ID
    
    Returns:
        是否成功接受撤单
    """
    target_id = order_or_id.order_id if isinstance(order_or_id, Order) else str(order_or_id)
    removed = False
    for idx, queued in list(enumerate(_order_queue)):
        if queued.order_id == target_id:
            _order_queue.pop(idx)
            log.info(f"🗑️ 本地队列撤单成功: {target_id}")
            removed = True
            break
    engine = get_current_engine()
    broker_id = None
    if isinstance(order_or_id, Order):
        broker_id = getattr(order_or_id, "_broker_order_id", None)
    if engine and getattr(engine, "broker", None) and broker_id:
        try:
            result = engine.broker.cancel_order(str(broker_id))
            if inspect.isawaitable(result):
                loop = getattr(engine, "_loop", None)
                if loop and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(result, loop)
                    result = fut.result()
                else:
                    result = asyncio.run(result)
            if result:
                log.info(f"🗑️ 券商撤单已提交: {broker_id}")
                return True
        except Exception as exc:
            log.warning(f"券商撤单失败 {broker_id}: {exc}")
    return removed


def cancel_all_orders() -> int:
    """取消本地队列所有订单，返回取消数量。"""
    count = len(_order_queue)
    _order_queue.clear()
    if count:
        log.info(f"🗑️ 已清空本地订单队列，共 {count} 笔")
    return count


def order_value(
    security: str,
    value: float,
    price: Optional[float] = None,
    style: Optional[Union[OrderStyle, MarketOrderStyle, LimitOrderStyle]] = None,
    wait_timeout: Optional[float] = None,
) -> Optional[Order]:
    """
    按价值下单
    
    Args:
        security: 标的代码
        value: 目标价值，正数表示买入，负数表示卖出
        price: 委托价格，None表示市价单
        
    Returns:
        Order对象，如果下单失败返回None
        
    Note:
        实际数量会在撮合时根据当前价格计算
    """
    if isinstance(price, (MarketOrderStyle, LimitOrderStyle)):
        style = price
        price = None

    if value == 0:
        log.warning(f"下单价值为0，忽略订单: {security}")
        return None
    
    # 临时订单，amount会在撮合时计算
    if style is not None:
        resolved_style: object = style
    elif price is not None:
        resolved_style = MarketOrderStyle(limit_price=price)
    else:
        resolved_style = MarketOrderStyle()

    order_obj = Order(
        order_id=_generate_order_id(),
        security=security,
        amount=0,  # 会在撮合时计算
        price=price if price is not None else 0.0,
        status=OrderStatus.open,
        add_time=datetime.now(),
        is_buy=(value > 0),
        style=resolved_style,
        wait_timeout=wait_timeout,
    )
    
    # 存储目标价值，用于撮合时计算
    order_obj._target_value = abs(value)  # type: ignore
    
    _order_queue.append(order_obj)
    log.debug(f"创建订单（按价值）: {security}, 价值: {value}")
    _trigger_order_processing(wait_timeout)
    
    return order_obj


def order_target(
    security: str,
    amount: int,
    price: Optional[float] = None,
    style: Optional[Union[OrderStyle, MarketOrderStyle, LimitOrderStyle]] = None,
    wait_timeout: Optional[float] = None,
) -> Optional[Order]:
    """
    目标股数下单（调整持仓到目标数量）
    """
    if isinstance(price, (MarketOrderStyle, LimitOrderStyle)):
        style = price
        price = None

    if style is not None:
        resolved_style: object = style
    elif price is not None:
        resolved_style = MarketOrderStyle(limit_price=price)
    else:
        resolved_style = MarketOrderStyle()

    order_obj = Order(
        order_id=_generate_order_id(),
        security=security,
        amount=abs(amount),
        price=price if price is not None else 0.0,
        status=OrderStatus.open,
        add_time=datetime.now(),
        is_buy=True,
        style=resolved_style,
        wait_timeout=wait_timeout,
    )

    order_obj._is_target_amount = True  # type: ignore
    order_obj._target_amount = amount  # type: ignore

    _order_queue.append(order_obj)
    log.debug(f"创建订单（目标股数）: {security}, 目标数量: {amount}")
    _trigger_order_processing(wait_timeout)

    return order_obj


def order_target_value(
    security: str,
    value: float,
    price: Optional[float] = None,
    style: Optional[Union[OrderStyle, MarketOrderStyle, LimitOrderStyle]] = None,
    wait_timeout: Optional[float] = None,
) -> Optional[Order]:
    """
    目标价值下单（调整持仓到目标价值）
    """
    if isinstance(price, (MarketOrderStyle, LimitOrderStyle)):
        style = price
        price = None

    if style is not None:
        resolved_style: object = style
    elif price is not None:
        resolved_style = MarketOrderStyle(limit_price=price)
    else:
        resolved_style = MarketOrderStyle()

    order_obj = Order(
        order_id=_generate_order_id(),
        security=security,
        amount=0,
        price=price if price is not None else 0.0,
        status=OrderStatus.open,
        add_time=datetime.now(),
        is_buy=True,
        style=resolved_style,
        wait_timeout=wait_timeout,
    )

    order_obj._is_target_value = True  # type: ignore
    order_obj._target_value = value  # type: ignore

    _order_queue.append(order_obj)
    log.debug(f"创建订单（目标价值）: {security}, 目标价值 {value}")
    _trigger_order_processing(wait_timeout)

    return order_obj


def get_order_queue():
    """获取当前订单队列"""
    return _order_queue


def clear_order_queue():
    """清空订单队列"""
    global _order_queue
    _order_queue = []


__all__ = [
    'order',
    'order_value',
    'order_target',
    'order_target_value',
    'get_order_queue',
    'clear_order_queue',
    'MarketOrderStyle',
    'LimitOrderStyle',
]
