from __future__ import annotations

import os
import sqlite3
from datetime import datetime, date as Date
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import logging

from .base import DataProvider
from ..cache import CacheManager

logger = logging.getLogger(__name__)


class LocalProvider(DataProvider):
    """
    基于 SQLite 数据库的本地数据提供者。
    数据通过 data/fetcher.py 预先保存到 SQLite 数据库中。
    """

    name: str = "local"
    requires_live_data: bool = False
    _SUFFIX_TO_JQ: Dict[str, str] = {
        "SZ": "XSHE",
        "SH": "XSHG",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        # 数据库路径配置优先级：
        # 1. 代码参数（config 中的 db_path）
        # 2. 环境变量 LOCAL_DB_PATH
        # 3. 默认路径：localdata/data/qmt_data.db（需要显式配置）
        db_path = (
            self.config.get("db_path") 
            or os.getenv("LOCAL_DB_PATH")
        )
        if db_path is None:
            raise ValueError("数据库路径未配置，无法初始化 LocalProvider")
        else:
            db_path = Path(db_path)
        
        self.db_path = str(db_path.resolve())  # 使用绝对路径
        self.config["db_path"] = self.db_path
        
        # 缓存配置（可选，本地数据通常不需要缓存）
        cache_dir_set = "cache_dir" in self.config
        cache_dir = self.config.get("cache_dir")
        self._cache = CacheManager(
            provider_name=self.name,
            cache_dir=cache_dir,
            fallback_to_env=not cache_dir_set,
        )
        
        # 验证数据库文件是否存在
        if not os.path.exists(self.db_path):
            logger.warning(f"本地数据库文件不存在: {self.db_path}")

    # ------------------------ 工具函数 ------------------------
    @classmethod
    def _normalize_security_code(cls, security: str) -> str:
        """将聚宽格式代码转换为 QMT 格式（用于数据库查询）"""
        if not security:
            return security
        sec = security.strip()
        if "." not in sec:
            return sec.upper()
        code, suffix = sec.split(".", 1)
        # XSHE -> SZ, XSHG -> SH
        if suffix.upper() == "XSHE":
            return f"{code.upper()}.SZ"
        elif suffix.upper() == "XSHG":
            return f"{code.upper()}.SH"
        return sec.upper()

    @classmethod
    def _to_jq_code(cls, qmt_code: str) -> str:
        """将 QMT 格式代码转换为聚宽格式"""
        if not qmt_code or "." not in qmt_code:
            return qmt_code
        code, suffix = qmt_code.split(".", 1)
        jq_suffix = cls._SUFFIX_TO_JQ.get(suffix.upper())
        if jq_suffix:
            return f"{code}.{jq_suffix}"
        return qmt_code

    @staticmethod
    def _to_date(value: Optional[Union[str, datetime, Date]]) -> Optional[Date]:
        """转换为日期对象"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, Date):
            return value
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None

    def _format_time(self, value: Optional[Union[str, datetime, Date]], period: str) -> str:
        """格式化时间为字符串"""
        if value is None:
            return ""
        dt = value
        if isinstance(value, str):
            dt = pd.to_datetime(value)
        if isinstance(dt, Date) and not isinstance(dt, datetime):
            dt = datetime.combine(dt, datetime.min.time())
        fmt = "%Y%m%d" if period == "1d" else "%Y%m%d%H%M%S"
        return dt.strftime(fmt)

    @staticmethod
    def _normalize_period(frequency: Optional[str]) -> str:
        """标准化频率"""
        if not frequency:
            return "1d"
        freq = str(frequency).strip().lower()
        alias = {
            "daily": "1d",
            "day": "1d",
            "1day": "1d",
            "d": "1d",
            "minute": "1m",
            "min": "1m",
            "1minute": "1m",
            "m": "1m",
        }
        normalized = alias.get(freq)
        if normalized:
            return normalized
        if freq.endswith(("m", "d")) and freq[:-1].isdigit():
            return freq
        return "1d"

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    # ------------------------ 认证 ------------------------
    def auth(self, user: Optional[str] = None, pwd: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """本地数据源无需认证"""
        pass

    # ------------------------ K 线数据 ------------------------
    def get_price(
        self,
        security: Union[str, List[str]],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        frequency: str = "daily",
        fields: Optional[List[str]] = None,
        skip_paused: bool = False,
        fq: str = "pre",
        count: Optional[int] = None,
        panel: bool = True,
        fill_paused: bool = True,
        pre_factor_ref_date: Optional[Union[str, datetime]] = None,
        prefer_engine: bool = False,
    ) -> pd.DataFrame:
        """从 SQLite 数据库获取价格数据"""
        securities = security if isinstance(security, (list, tuple)) else [security]
        period = self._normalize_period(frequency)
        
        # 目前只支持日线数据
        if period != "1d":
            logger.warning(f"LocalProvider 目前只支持日线数据，请求的频率 {frequency} 将被忽略")
            period = "1d"
        
        frames: Dict[str, pd.DataFrame] = {}
        
        for sec in securities:
            kwargs = {
                "security": sec,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": period,
                "fields": fields,
                "skip_paused": skip_paused,
                "fq": fq,
                "count": count,
            }
            
            def _fetch_single(kw: Dict[str, Any]) -> pd.DataFrame:
                return self._get_price_single(
                    kw["security"],
                    start_date=kw.get("start_date"),
                    end_date=kw.get("end_date"),
                    frequency=kw.get("frequency", "1d"),
                    fields=kw.get("fields"),
                    skip_paused=kw.get("skip_paused", False),
                    fq=kw.get("fq", "pre"),
                    count=kw.get("count"),
                )
            
            frames[sec] = self._cache.cached_call("get_price", kwargs, _fetch_single, result_type="df")
        
        if len(frames) == 1:
            return next(iter(frames.values()))
        if panel:
            return pd.concat(frames, axis=1)
        
        rows = []
        for sec, df in frames.items():
            tmp = df.copy()
            tmp["code"] = sec
            rows.append(tmp)
        return pd.concat(rows, axis=0)

    def _get_price_single(
        self,
        security: str,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]],
        frequency: str,
        fields: Optional[List[str]],
        skip_paused: bool,
        fq: Optional[str],
        count: Optional[int],
    ) -> pd.DataFrame:
        """获取单个标的价格数据"""
        conn = self._get_connection()
        try:
            # 转换为 QMT 格式代码用于查询
            qmt_code = self._normalize_security_code(security)
            
            # 构建 SQL 查询
            query = "SELECT date, time, open, high, low, close, volume, amount as money, preClose, suspendFlag FROM daily_kline WHERE code = ?"
            params = [qmt_code]
            
            # 添加日期过滤
            if start_date:
                start_str = self._format_time(start_date, "1d")
                query += " AND date >= ?"
                params.append(start_str)
            
            if end_date:
                end_str = self._format_time(end_date, "1d")
                query += " AND date <= ?"
                params.append(end_str)
            
            query += " ORDER BY date ASC"
            
            # 执行查询
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                return pd.DataFrame()
            
            # 转换日期列为索引
            # 数据库中的 date 可能是字符串格式（YYYYMMDD）或日期格式
            try:
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            except Exception:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])  # 删除无法解析的日期
            if df.empty:
                return pd.DataFrame()
            df.set_index("date", inplace=True)
            df.index.name = None
            
            # 处理停牌标志
            if "suspendFlag" in df.columns:
                df["paused"] = (df["suspendFlag"] == 1).astype(float)
                df.drop(columns=["suspendFlag"], inplace=True)
            else:
                df["paused"] = 0.0
            
            # 处理停牌数据
            if skip_paused:
                df = df[df.get("volume", 0) > 0]
            
            # 处理 count 参数
            if count:
                df = df.tail(count)
            
            # 处理字段过滤
            if fields:
                missing = [f for f in fields if f not in df.columns]
                for f in missing:
                    df[f] = 0.0
                df = df[fields]
            
            # 注意：本地数据库存储的是前复权数据（front_ratio），
            # 如果请求的是后复权或不复权，这里暂时返回前复权数据
            # 可以根据需要扩展支持
            
            return df
            
        except Exception as e:
            logger.error(f"从数据库获取 {security} 数据失败: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    # ------------------------ 交易日/基础信息 ------------------------
    def get_trade_days(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        count: Optional[int] = None,
    ) -> List[datetime]:
        """从数据库获取交易日列表"""
        kwargs = {"start_date": start_date, "end_date": end_date, "count": count}
        
        def _fetch(kw: Dict[str, Any]) -> List[datetime]:
            conn = self._get_connection()
            try:
                query = "SELECT DISTINCT date FROM daily_kline WHERE 1=1"
                params = []
                
                if kw.get("start_date"):
                    start_str = self._format_time(kw.get("start_date"), "1d")
                    query += " AND date >= ?"
                    params.append(start_str)
                
                if kw.get("end_date"):
                    end_str = self._format_time(kw.get("end_date"), "1d")
                    query += " AND date <= ?"
                    params.append(end_str)
                
                query += " ORDER BY date ASC"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if df.empty:
                    return []
                
                # 处理日期格式
                try:
                    dates = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
                except Exception:
                    dates = pd.to_datetime(df["date"], errors="coerce")
                dates = dates.dropna().tolist()
                
                # 处理 count 参数
                if kw.get("count"):
                    if kw.get("count") > 0:
                        dates = dates[-kw.get("count"):]
                    else:
                        dates = dates[:abs(kw.get("count"))]
                
                return [d.to_pydatetime() if hasattr(d, "to_pydatetime") else d for d in dates]
                
            except Exception as e:
                logger.error(f"获取交易日失败: {e}")
                return []
            finally:
                conn.close()
        
        return self._cache.cached_call("get_trade_days", kwargs, _fetch, result_type="list_date")

    def get_all_securities(
        self,
        types: Union[str, List[str]] = "stock",
        date: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """从数据库获取所有标的信息"""
        if isinstance(types, str):
            types = [types]
        kwargs = {"types": tuple(sorted(types)), "date": date}
        
        def _fetch(kw: Dict[str, Any]) -> Dict[str, Any]:
            conn = self._get_connection()
            try:
                rows = []
                
                for t in kw["types"]:
                    table_name = None
                    if t == "stock":
                        table_name = "stocks"
                    elif t == "index":
                        table_name = "indices"
                    elif t == "etf":
                        table_name = "etfs"
                    
                    if not table_name:
                        continue
                    
                    query = f"SELECT code, name, display_name, market, start_date, end_date, type, subtype, parent FROM {table_name}"
                    df = pd.read_sql_query(query, conn)
                    
                    for _, row in df.iterrows():
                        qmt_code = row["code"]
                        jq_code = self._to_jq_code(qmt_code)
                        # 解析 start_date
                        start_date = None
                        if pd.notna(row.get("start_date")):
                            try:
                                start_date = pd.to_datetime(row.get("start_date"), errors="coerce")
                            except Exception:
                                pass
                        # 解析 end_date
                        end_date = None
                        if pd.notna(row.get("end_date")):
                            try:
                                end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
                            except Exception:
                                pass
                        # 如果没有 end_date，设置为 2200-01-01
                        if end_date is None or pd.isna(end_date):
                            end_date = pd.to_datetime('2200-01-01')
                        
                        rows.append({
                            "ts_code": jq_code,
                            "display_name": row.get("display_name") or row.get("name", qmt_code),
                            "name": row.get("name") or (qmt_code.split(".", 1)[0] if "." in qmt_code else qmt_code),
                            "start_date": start_date,
                            "end_date": end_date,
                            "type": row.get("type") or t,
                        })
                
                if not rows:
                    return {}
                
                df = pd.DataFrame(rows).drop_duplicates("ts_code").set_index("ts_code")
                return df.to_dict(orient="index")
                
            except Exception as e:
                logger.error(f"获取标的信息失败: {e}")
                return {}
            finally:
                conn.close()
        
        data = self._cache.cached_call("get_all_securities", kwargs, _fetch, result_type="list_dict")
        if not data:
            return pd.DataFrame(columns=["display_name", "name", "start_date", "end_date", "type"])
        df = pd.DataFrame.from_dict(data, orient="index")
        if not df.empty:
            df["qmt_code"] = [self._normalize_security_code(code) for code in df.index]
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    def get_index_stocks(self, index_symbol: str, date: Optional[Union[str, datetime]] = None) -> List[str]:
        """获取指数成分股（本地数据库暂不支持，返回空列表）"""
        logger.warning("LocalProvider 暂不支持获取指数成分股")
        return []

    def get_security_info(self, security: str) -> Dict[str, Any]:
        """获取标的详细信息"""
        conn = self._get_connection()
        try:
            qmt_code = self._normalize_security_code(security)
            jq_code = self._to_jq_code(qmt_code)
            
            # 尝试从各个表中查找
            table_type_map = {
                "stocks": "stock",
                "indices": "index",
                "etfs": "etf",
            }
            for table, sec_type in table_type_map.items():
                query = f"SELECT code, name, display_name, market, start_date, end_date, type, subtype, parent FROM {table} WHERE code = ?"
                df = pd.read_sql_query(query, conn, params=[qmt_code])
                if not df.empty:
                    row = df.iloc[0]
                    # 解析 start_date
                    start_date = None
                    if pd.notna(row.get("start_date")):
                        try:
                            start_date = pd.to_datetime(row.get("start_date"), errors="coerce")
                            if pd.notna(start_date):
                                start_date = start_date.date()
                        except Exception:
                            pass
                    # 解析 end_date
                    end_date = None
                    if pd.notna(row.get("end_date")):
                        try:
                            end_date = pd.to_datetime(row.get("end_date"), errors="coerce")
                            if pd.notna(end_date):
                                end_date = end_date.date()
                        except Exception:
                            pass
                    # 如果没有 end_date，设置为 2200-01-01
                    if end_date is None:
                        from datetime import date as Date
                        end_date = Date(2200, 1, 1)
                    
                    return {
                        "display_name": row.get("display_name") or row.get("name", jq_code),
                        "name": row.get("name") or (qmt_code.split(".", 1)[0] if "." in qmt_code else qmt_code),
                        "start_date": start_date,
                        "end_date": end_date,
                        "type": row.get("type") or sec_type,
                        "subtype": row.get("subtype"),
                        "parent": row.get("parent"),
                    }
            
            # 未找到，返回默认值
            return {
                "display_name": jq_code,
                "name": jq_code,
                "start_date": None,
                "end_date": None,
                "type": "stock",
                "subtype": None,
                "parent": None,
            }
        except Exception as e:
            logger.error(f"获取 {security} 信息失败: {e}")
            return {
                "display_name": security,
                "name": security,
                "start_date": None,
                "end_date": None,
                "type": "stock",
                "subtype": None,
                "parent": None,
            }
        finally:
            conn.close()

    # ------------------------ 分红 / 拆分 ------------------------
    def get_split_dividend(
        self,
        security: str,
        start_date: Optional[Union[str, datetime, Date]] = None,
        end_date: Optional[Union[str, datetime, Date]] = None,
    ) -> List[Dict[str, Any]]:
        """获取分红拆分数据（本地数据库暂不支持，返回空列表）"""
        logger.warning("LocalProvider 暂不支持获取分红拆分数据")
        return []

