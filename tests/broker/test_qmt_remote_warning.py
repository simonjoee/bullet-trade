import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bullet_trade.broker.qmt_remote import RemoteQmtBroker


class _FakeConn:
    def __init__(self):
        self.requests = []

    def start(self):
        pass

    def close(self):
        pass

    def request(self, action, payload, timeout=30.0):
        self.requests.append((action, payload))
        return {"order_id": "oid-1", "warning": "000001.XSHE 停牌，拒绝远程委托"}


def test_remote_warning_prints_and_captures(capsys, monkeypatch):
    monkeypatch.setenv("QMT_SERVER_TOKEN", "dummy-token")
    broker = RemoteQmtBroker(account_id="acc")
    broker._connection = _FakeConn()  # type: ignore
    broker.connect()
    broker._place_order_sync("BUY", "000001.XSHE", 100, None, None)
    out = capsys.readouterr().out
    assert "停牌" in out
    assert broker._last_warning and "停牌" in broker._last_warning
