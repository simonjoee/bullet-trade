import pytest

from bullet_trade.broker.qmt import QmtBroker
from bullet_trade.core.models import OrderStatus


@pytest.mark.unit
def test_qmt_broker_get_orders_filters(monkeypatch):
    broker = QmtBroker(account_id="demo")
    broker._connected = True

    monkeypatch.setattr(
        broker,
        "sync_orders",
        lambda: [
            {"order_id": "1", "security": "000001.XSHE", "status": "open"},
            {"order_id": "2", "security": "000002.XSHE", "status": "filled"},
        ],
    )

    orders = broker.get_orders(status=OrderStatus.open)
    ids = {item.get("order_id") for item in orders}
    assert "1" in ids
    assert "2" not in ids


@pytest.mark.unit
def test_qmt_broker_get_trades_mapping(monkeypatch):
    broker = QmtBroker(account_id="demo")
    broker._connected = True

    class DummyTrade:
        def __init__(self):
            self.trade_id = "t1"
            self.order_id = "o1"
            self.stock_code = "000001.SZ"
            self.trade_volume = 100
            self.trade_price = 10.0
            self.trade_time = "2025-01-01 09:31:00"

    class DummyTrader:
        def query_stock_trades(self, account):
            return [DummyTrade()]

    broker._xt_trader = DummyTrader()
    broker._xt_account = object()

    trades = broker.get_trades(order_id="o1")
    assert len(trades) == 1
    trade = trades[0]
    assert trade.get("trade_id") == "t1"
    assert trade.get("order_id") == "o1"
    assert trade.get("security") == "000001.XSHE"
