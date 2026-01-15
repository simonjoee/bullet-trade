import datetime

from bullet_trade.core.engine import BacktestEngine
from bullet_trade.core.models import Order, OrderStatus, Trade


def _dummy_initialize(context):
    return None


def _dummy_handle_data(context, data):
    return None


def test_backtest_engine_order_trade_queries():
    engine = BacktestEngine(initialize=_dummy_initialize, handle_data=_dummy_handle_data)
    order = Order(
        order_id="o1",
        security="000001.XSHE",
        amount=100,
        price=10.0,
        status=OrderStatus.open,
        add_time=datetime.datetime.now(),
        is_buy=True,
    )
    engine._register_order(order)

    orders = engine.get_orders()
    assert "o1" in orders
    open_orders = engine.get_open_orders()
    assert "o1" in open_orders

    trade = Trade(
        order_id="o1",
        security="000001.XSHE",
        amount=100,
        price=10.0,
        time=datetime.datetime.now(),
        trade_id="t1",
    )
    engine.trades.append(trade)
    trades = engine.get_trades(order_id="o1")
    assert "t1" in trades
