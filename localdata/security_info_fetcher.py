"""
证券基本信息获取器

负责从 QMT 获取股票、指数、ETF 的基本信息并保存到数据库
"""

import sqlite3
import pandas as pd
from xtquant import xtdata


class SecurityInfoFetcher:
    def __init__(self, db_path=None):
        from pathlib import Path
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "qmt_data.db"
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库表结构（仅基本信息表）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建股票信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT,
                display_name TEXT,
                market TEXT,
                start_date TEXT,
                end_date TEXT,
                type TEXT,
                subtype TEXT,
                parent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建指数信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT,
                display_name TEXT,
                market TEXT,
                start_date TEXT,
                end_date TEXT,
                type TEXT,
                subtype TEXT,
                parent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建ETF信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS etfs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                name TEXT,
                display_name TEXT,
                market TEXT,
                start_date TEXT,
                end_date TEXT,
                type TEXT,
                subtype TEXT,
                parent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print("证券基本信息表初始化完成")

    def _parse_instrument_info(self, info):
        """解析 QMT 的 instrument_detail 信息，返回标准化的字段"""
        # 获取基本信息
        display_name = info.get('InstrumentName', '') or ''
        name = info.get('InstrumentID', '') or info.get('InstrumentCode', '') or ''
        market = info.get('ExchangeID', '') or ''
        
        # 解析上市日期（OpenDate）
        open_date = info.get('OpenDate', '')
        start_date = None
        if open_date:
            try:
                if isinstance(open_date, str):
                    if len(open_date) == 8 and open_date.isdigit():
                        start_date = f"{open_date[:4]}-{open_date[4:6]}-{open_date[6:8]}"
                    else:
                        pd_date = pd.to_datetime(open_date, errors='coerce')
                        if pd.notna(pd_date):
                            start_date = pd_date.strftime('%Y-%m-%d')
            except Exception:
                pass
        
        # 解析退市日期（ExpireDate）
        expire_date = info.get('ExpireDate', '')
        end_date = None
        if expire_date:
            try:
                if isinstance(expire_date, str):
                    if len(expire_date) == 8 and expire_date.isdigit():
                        end_date = f"{expire_date[:4]}-{expire_date[4:6]}-{expire_date[6:8]}"
                    else:
                        pd_date = pd.to_datetime(expire_date, errors='coerce')
                        if pd.notna(pd_date):
                            end_date = pd_date.strftime('%Y-%m-%d')
            except Exception:
                pass
        
        # 如果没有退市日期，设置为 2200-01-01（表示未退市）
        if not end_date:
            end_date = '2200-01-01'
        
        # 判断类型和子类型
        # 根据 QMT 的字段判断，这里需要根据实际情况调整
        sec_type = info.get('ProductType', 'stock')  # 可能需要根据实际情况调整
        subtype = info.get('ProductSubType', None)
        
        # 获取分级基金的母基金代码
        parent = info.get('ParentCode', None) or info.get('ParentInstrumentID', None)
        
        return {
            'display_name': display_name,
            'name': name,
            'market': market,
            'start_date': start_date,
            'end_date': end_date,
            'type': sec_type,
            'subtype': subtype,
            'parent': parent,
        }

    def save_stock_info(self, stock_symbols):
        """保存股票基本信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol in stock_symbols:
            try:
                # 获取股票基本信息
                info = xtdata.get_instrument_detail(symbol)
                parsed = self._parse_instrument_info(info)
                # 股票类型固定为 'stock'
                parsed['type'] = 'stock'

                cursor.execute('''
                    INSERT OR REPLACE INTO stocks 
                    (code, name, display_name, market, start_date, end_date, type, subtype, parent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    parsed['name'], 
                    parsed['display_name'],
                    parsed['market'], 
                    parsed['start_date'],
                    parsed['end_date'],
                    parsed['type'],
                    parsed['subtype'],
                    parsed['parent']
                ))

            except Exception as e:
                print(f"保存股票 {symbol} 信息时出错: {e}")

        conn.commit()
        conn.close()
        print("股票基本信息保存完成")

    def save_index_info(self, index_symbols):
        """保存指数基本信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol in index_symbols:
            try:
                # 获取指数基本信息
                info = xtdata.get_instrument_detail(symbol)
                parsed = self._parse_instrument_info(info)
                # 指数类型固定为 'index'
                parsed['type'] = 'index'

                cursor.execute('''
                    INSERT OR REPLACE INTO indices 
                    (code, name, display_name, market, start_date, end_date, type, subtype, parent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    parsed['name'], 
                    parsed['display_name'],
                    parsed['market'], 
                    parsed['start_date'],
                    parsed['end_date'],
                    parsed['type'],
                    parsed['subtype'],
                    parsed['parent']
                ))

            except Exception as e:
                print(f"保存指数 {symbol} 信息时出错: {e}")

        conn.commit()
        conn.close()
        print("指数基本信息保存完成")

    def save_etf_info(self, symbols):
        """保存ETF基本信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for symbol in symbols:
            try:
                # 获取ETF基本信息
                info = xtdata.get_instrument_detail(symbol)
                parsed = self._parse_instrument_info(info)
                # ETF类型固定为 'etf'，但也可以根据实际情况判断是否为其他基金类型
                if not parsed['type'] or parsed['type'] == 'stock':
                    parsed['type'] = 'etf'

                cursor.execute('''
                    INSERT OR REPLACE INTO etfs 
                    (code, name, display_name, market, start_date, end_date, type, subtype, parent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, 
                    parsed['name'], 
                    parsed['display_name'],
                    parsed['market'], 
                    parsed['start_date'],
                    parsed['end_date'],
                    parsed['type'],
                    parsed['subtype'],
                    parsed['parent']
                ))

            except Exception as e:
                print(f"保存ETF {symbol} 信息时出错: {e}")

        conn.commit()
        conn.close()
        print("ETF 基本信息保存完成")

    def fetch_and_save_stock_info(self):
        """获取并保存股票基本信息"""
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        self.save_stock_info(stocks)
        return stocks

    def fetch_and_save_index_info(self):
        """获取并保存指数基本信息"""
        indexes = xtdata.get_stock_list_in_sector('沪深指数')
        self.save_index_info(indexes)
        return indexes

    def fetch_and_save_etf_info(self):
        """获取并保存 ETF 基本信息"""
        etfs = xtdata.get_stock_list_in_sector('沪深ETF')
        self.save_etf_info(etfs)
        return etfs


# 使用示例
if __name__ == "__main__":
    # 创建证券信息获取器实例
    fetcher = SecurityInfoFetcher()

    # 获取并保存股票信息
    # stock_list = fetcher.fetch_and_save_stock_info()

    # 获取并保存指数信息
    # index_list = fetcher.fetch_and_save_index_info()

    # 获取并保存ETF信息
    etf_list = fetcher.fetch_and_save_etf_info()

    print("证券基本信息获取任务完成")

