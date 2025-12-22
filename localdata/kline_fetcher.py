"""
K线数据获取器

负责从 QMT 获取每日K线数据并保存到数据库
"""

import sqlite3
import pandas as pd
from xtquant import xtdata


class KlineFetcher:
    def __init__(self, db_path=None):
        from pathlib import Path
        if db_path is None:
            db_path = Path(__file__).parent / "data" / "qmt_data.db"
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库表结构（仅K线数据表）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建日线数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_kline (
                code TEXT NOT NULL,                 -- 股票代码
                date TEXT NOT NULL,                 -- 日期
                time INTEGER,                       -- 时间
                open FLOAT,                         -- 开盘价
                high FLOAT,                         -- 最高价
                low FLOAT,                          -- 最低价
                close FLOAT,                        -- 收盘价
                volume FLOAT,                       -- 成交量
                amount FLOAT,                       -- 成交额
                openInterest FLOAT,                 -- 持仓量
                preClose FLOAT,                     -- 前收盘价
                suspendFlag INT,                    -- 停牌 1停牌，0 不停牌
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (code, date)
            )
        ''')

        # 创建索引提高查询性能
        # 单独为date创建索引以支持按日期查询的需求
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_kline(date)')

        conn.commit()
        conn.close()
        print("K线数据表初始化完成")

    def fetch_and_save_daily_data(self, symbols, start_date="", end_date=""):
        """
        获取并保存日线数据
        
        Args:
            symbols: 标的代码列表
            start_date: 开始日期，格式如 "20251101" 或 "2025-11-01"
            end_date: 结束日期，格式如 "20251101" 或 "2025-11-01"
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        success_count = 0
        fails = []

        for i, symbol in enumerate(symbols):
            try:
                # 下载日线数据
                xtdata.download_history_data(symbol, period='1d', incrementally=True)

                data = xtdata.get_market_data_ex(
                    stock_list=[symbol],
                    start_time=start_date,
                    end_time=end_date,
                    count=0,  # 获取所有数据
                    dividend_type='front_ratio'
                )

                if data is None:
                    fails.append(symbol)
                    continue
                    
                df = pd.DataFrame(data[symbol])
                if df.empty:
                    fails.append(symbol)
                    continue

                # 添加股票代码列
                df['code'] = symbol
                # 重置索引，将时间作为列
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'date'}, inplace=True)

                # 插入数据库
                for _, row in df.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_kline 
                        (code, date, time, open, high, low, close, volume, amount, openInterest, preClose, suspendFlag)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['code'], row['date'], 
                        row.get('time'), 
                        row.get('open'), row.get('high'),
                        row.get('low'), row.get('close'), 
                        row.get('volume'), row.get('amount'),
                        row.get('openInterest'),
                        row.get('preClose'), 
                        row.get('suspendFlag')
                    ))

                success_count += 1
                if success_count % 50 == 0:
                    print(f"已处理 {success_count} 个标的")

            except Exception as e:
                fails.append(symbol)
                print(f"处理 {symbol} 时出错: {e}")

        conn.commit()
        conn.close()
        print(f"K线数据获取完成: 成功 {success_count} 个，失败 {len(fails)} 个")
        if fails:
            print(f"失败symbols: {fails}")

    def fetch_and_save_stock_daily_data(self, start_date="", end_date=""):
        """获取并保存所有股票的日线数据"""
        stocks = xtdata.get_stock_list_in_sector('沪深A股')
        self.fetch_and_save_daily_data(stocks, start_date, end_date)

    def fetch_and_save_index_daily_data(self, start_date="", end_date=""):
        """获取并保存所有指数的日线数据"""
        indexes = xtdata.get_stock_list_in_sector('沪深指数')
        self.fetch_and_save_daily_data(indexes, start_date, end_date)

    def fetch_and_save_etf_daily_data(self, start_date="", end_date=""):
        """获取并保存所有ETF的日线数据"""
        etfs = xtdata.get_stock_list_in_sector('沪深ETF')
        self.fetch_and_save_daily_data(etfs, start_date, end_date)


# 使用示例
if __name__ == "__main__":
    # 创建K线数据获取器实例
    fetcher = KlineFetcher()

    # 获取并保存股票的日线数据
    # fetcher.fetch_and_save_stock_daily_data("20251101")

    # 获取并保存指数的日线数据
    # fetcher.fetch_and_save_index_daily_data("20251101")

    # 获取并保存ETF的日线数据
    fetcher.fetch_and_save_etf_daily_data("20001101")

    # 或者指定标的列表
    # symbol_list = ['000001.SZ', '000002.SZ']
    # fetcher.fetch_and_save_daily_data(symbol_list, "20251101")

    print("K线数据获取任务完成")

