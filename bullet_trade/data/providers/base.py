from abc import ABC, abstractmethod
from typing import Union, List, Optional, Dict, Any
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    """
    抽象数据提供者接口。
    不同数据源（jqdatasdk、tushare、miniqmt等）实现该接口，
    以便在框架内可插拔切换数据来源。
    """
    name: str = "base"
    # 是否要求实时行情必须由 provider 提供（用于 live 模式防止回落到历史数据）
    requires_live_data: bool = False

    def auth(self, user: Optional[str] = None, pwd: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """执行数据源认证或初始化。可选。默认读取环境变量，也可传入账号参数。"""
        pass

    @abstractmethod
    def get_price(
        self,
        security: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        frequency: str = 'daily',
        fields: Optional[List[str]] = None,
        skip_paused: bool = False,
        fq: str = 'pre',
        count: Optional[int] = None,
        panel: bool = True,
        fill_paused: bool = True,
        pre_factor_ref_date: Optional[Union[str, datetime]] = None,
        prefer_engine: bool = False,
    ) -> pd.DataFrame:
        pass


    @abstractmethod
    def get_trade_days(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        count: Optional[int] = None
    ) -> List[datetime]:
        pass

    @abstractmethod
    def get_all_securities(
        self,
        types: Union[str, List[str]] = 'stock',
        date: Optional[Union[str, datetime]] = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_index_stocks(
        self,
        index_symbol: str,
        date: Optional[Union[str, datetime]] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def get_split_dividend(
        self,
        security: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        pass

    def get_security_info(self, security: str) -> Dict[str, Any]:
        """
        返回指定标的的元信息，例如类型或子类型。
        默认返回空字典，可由具体数据提供者覆盖。
        """
        return {}

    def get_current_tick(self, security: str) -> Optional[Dict[str, Any]]:
        """
        返回最新 tick 快照（可选实现）。

        若数据源不支持实时 tick，可直接返回 None；
        支持实时行情的数据源（如 miniQMT/xtdata）可覆盖该方法以提供 sid/last_price/dt 等字段。
        """
        return None

    def subscribe_ticks(self, symbols: List[str]) -> None:
        """
        订阅指定标的 tick（可选实现）。默认不操作。
        """
        return None

    def subscribe_markets(self, markets: List[str]) -> None:
        """
        订阅市场级 tick（可选实现，如 ['SH','SZ']）。默认不操作。
        """
        return None

    def unsubscribe_ticks(self, symbols: Optional[List[str]] = None) -> None:
        """
        取消 tick 订阅（可选实现）。symbols 为 None 表示全部取消。默认不操作。
        """
        return None

    def unsubscribe_markets(self, markets: Optional[List[str]] = None) -> None:
        """
        取消市场级 tick 订阅（可选实现）。默认不操作。
        """
        return None
