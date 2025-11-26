import datetime as dt

import pandas as pd
import pytest

import bullet_trade.data.providers.jqdata as jqdata
from bullet_trade.data.providers.jqdata import JQDataProvider


@pytest.mark.unit
class TestJQDataProviderAuth:
    def test_auth_prefers_new_env_names(self, monkeypatch):
        provider = JQDataProvider()
        recorded = {}

        def fake_auth(user, pwd, host=None, port=None):
            payload = {'user': user, 'pwd': pwd}
            if host is not None:
                payload['host'] = host
            if port is not None:
                payload['port'] = port
            recorded.update(payload)

        monkeypatch.setattr(jqdata.jq, 'auth', fake_auth)

        for legacy_key in ('JQDATA_USER', 'JQDATA_PWD'):
            monkeypatch.delenv(legacy_key, raising=False)

        monkeypatch.setenv('JQDATA_USERNAME', 'new_user')
        monkeypatch.setenv('JQDATA_PASSWORD', 'new_pwd')
        monkeypatch.setenv('JQDATA_SERVER', 'srv.test')
        monkeypatch.setenv('JQDATA_PORT', '1234')

        provider.auth()

        assert recorded == {
            'user': 'new_user',
            'pwd': 'new_pwd',
            'host': 'srv.test',
            'port': 1234,
        }

    def test_auth_falls_back_to_legacy_env_names(self, monkeypatch):
        provider = JQDataProvider()
        recorded = {}

        def fake_auth(user, pwd, host=None, port=None):
            payload = {'user': user, 'pwd': pwd}
            if host is not None:
                payload['host'] = host
            if port is not None:
                payload['port'] = port
            recorded.update(payload)

        monkeypatch.setattr(jqdata.jq, 'auth', fake_auth)

        for new_key in ('JQDATA_USERNAME', 'JQDATA_PASSWORD'):
            monkeypatch.delenv(new_key, raising=False)

        monkeypatch.setenv('JQDATA_USER', 'legacy_user')
        monkeypatch.setenv('JQDATA_PWD', 'legacy_pwd')
        monkeypatch.setenv('JQDATA_SERVER', '')
        monkeypatch.setenv('JQDATA_PORT', '')

        provider.auth()

        assert recorded == {
            'user': 'legacy_user',
            'pwd': 'legacy_pwd',
        }

    def test_auth_strips_inline_comments_from_port(self, monkeypatch):
        provider = JQDataProvider()
        recorded = {}

        def fake_auth(user, pwd, host=None, port=None):
            payload = {'user': user, 'pwd': pwd}
            if host is not None:
                payload['host'] = host
            if port is not None:
                payload['port'] = port
            recorded.update(payload)

        monkeypatch.setattr(jqdata.jq, 'auth', fake_auth)

        monkeypatch.setenv('JQDATA_USERNAME', 'comment_user')
        monkeypatch.setenv('JQDATA_PASSWORD', 'comment_pwd')
        monkeypatch.setenv('JQDATA_SERVER', 'srv.test')
        monkeypatch.setenv('JQDATA_PORT', '8087                   # 可选：端口')

        provider.auth()

        assert recorded == {
            'user': 'comment_user',
            'pwd': 'comment_pwd',
            'host': 'srv.test',
            'port': 8087,
        }

    def test_auth_strips_inline_comments_from_host(self, monkeypatch):
        provider = JQDataProvider()
        recorded = {}

        def fake_auth(user, pwd, host=None, port=None):
            payload = {'user': user, 'pwd': pwd}
            if host is not None:
                payload['host'] = host
            if port is not None:
                payload['port'] = port
            recorded.update(payload)

        monkeypatch.setattr(jqdata.jq, 'auth', fake_auth)

        monkeypatch.setenv('JQDATA_USERNAME', 'host_user')
        monkeypatch.setenv('JQDATA_PASSWORD', 'host_pwd')

        provider.auth()

        assert recorded == {
            'user': 'host_user',
            'pwd': 'host_pwd',
        }


class _DummyQuery:
    def filter(self, *args, **kwargs):
        return self


@pytest.mark.unit
def test_get_split_dividend_prefers_ratio_fields(monkeypatch):
    provider = JQDataProvider({'cache_dir': None})
    monkeypatch.setattr(JQDataProvider, '_infer_security_type', lambda self, *args, **kwargs: 'stock')
    monkeypatch.setattr(jqdata, 'query', lambda *args, **kwargs: _DummyQuery())

    df = pd.DataFrame(
        [
            {
                'a_xr_date': dt.date(2015, 7, 27),
                'bonus_ratio_rmb': 5.0,
                'dividend_ratio': 2.0,
                'transfer_ratio': 1.0,
            }
        ]
    )
    monkeypatch.setattr(jqdata.finance, 'run_query', lambda q: df)

    events = provider.get_split_dividend('601318.XSHG', start_date='2015-07-27', end_date='2015-07-27')

    assert len(events) == 1
    event = events[0]
    assert event['bonus_pre_tax'] == pytest.approx(5.0)
    assert event['scale_factor'] == pytest.approx(1.3)
    assert event['per_base'] == 10


@pytest.mark.unit
def test_get_split_dividend_falls_back_to_share_base(monkeypatch):
    provider = JQDataProvider({'cache_dir': None})
    monkeypatch.setattr(JQDataProvider, '_infer_security_type', lambda self, *args, **kwargs: 'stock')
    monkeypatch.setattr(jqdata, 'query', lambda *args, **kwargs: _DummyQuery())

    df = pd.DataFrame(
        [
            {
                'a_xr_date': dt.date(2015, 7, 27),
                'bonus_ratio_rmb': 4.0,
                'dividend_number': 30000.0,
                'transfer_number': 10000.0,
                'distributed_share_base_implement': 100000.0,
            }
        ]
    )
    monkeypatch.setattr(jqdata.finance, 'run_query', lambda q: df)

    events = provider.get_split_dividend('601318.XSHG', start_date='2015-07-27', end_date='2015-07-27')

    assert len(events) == 1
    event = events[0]
    assert event['bonus_pre_tax'] == pytest.approx(4.0)
    assert event['scale_factor'] == pytest.approx(1.4)
