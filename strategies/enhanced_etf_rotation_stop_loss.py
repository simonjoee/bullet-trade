# 基于"22[Z]带上止损的核心资产轮动"策略的简化版本
# 
# 策略特点：
# 1. 核心资产ETF轮动：通过动量因子选择表现最好的ETF
# 2. 止损保护：单日止损和回撤止损
# 3. 市场择时：市场极端情况下可空仓
#
# 简化说明：
# - 保留ETF动量轮动（核心功能）
# - 保留基本止损机制（单日止损、回撤止损）
# - 简化仓位管理（固定仓位比例）
# - 简化市场择时（基本判断）
#
# 克隆自聚宽文章：https://www.joinquant.com/post/47966
# 标题：带上止损的核心资产轮动才更安心，低回撤，高收益率
# 作者：养家大哥

import numpy as np
import pandas as pd
import math

# ========== 常量定义 ==========
MARKET_INDEX = '000300.XSHG'  # 市场指标（沪深300）
HISTORY_DAYS = 100  # ETF历史数据保留天数
MAX_HISTORY_LENGTH = 100  # 最大历史数据长度


def initialize(context):
    """初始化函数：设置策略参数和运行环境"""
    # 基础设置
    set_benchmark(MARKET_INDEX)
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.000))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, 
                             open_commission=0.0002, 
                             close_commission=0.0002, 
                             close_today_commission=0, 
                             min_commission=5), type='fund')
    log.set_level('system', 'error')
    
    # ========== 策略参数 ==========
    g.etf_pool = [
        '518880.XSHG',  # 黄金ETF
        '513100.XSHG',  # 纳指100
        '159915.XSHE',  # 创业板100
        '510180.XSHG',  # 上证180
        '510300.XSHG',  # 沪深300ETF
        '159919.XSHE',  # 沪深300ETF（深市）
    ]
    
    # 动量参数
    g.m_days = 25  # 动量参考天数
    
    # 止损参数
    g.daily_stop_loss = 0.96  # 单日止损线（跌幅超过4%）
    g.drawdown_stop_loss = 0.92  # 回撤止损线（回撤超过8%）
    
    # 择时参数
    g.market_timing = True
    g.market_ma_days = 20
    g.market_threshold = -0.03  # 市场跌幅阈值（-3%）
    
    # 仓位参数
    g.position_ratio = 0.8  # 固定仓位比例（80%）
    
    # 初始化数据
    g.etf_info = _init_etf_history(g.etf_pool, HISTORY_DAYS)
    g.etf_pre = None  # 上次选择的ETF
    g.etf_high_price = {}  # 记录每个ETF的最高价
    
    # 定时任务
    run_daily(update_etf_table, '7:00')
    run_daily(trade, '9:30')


def _init_etf_history(etf_pool, ndays):
    """初始化ETF历史价格数据"""
    etf_info = {}
    for etf in etf_pool:
        try:
            df = attribute_history(etf, ndays, '1d', ['close'], 
                                   skip_paused=True, df=True, fq='pre')
            etf_info[etf] = list(df.close)
        except Exception as e:
            log.error(f"初始化ETF {etf} 失败: {e}")
            etf_info[etf] = []
    return etf_info


def update_etf_table(context):
    """更新ETF价格数据"""
    for etf in g.etf_pool:
        try:
            df = attribute_history(etf, 1, '1d', ['close'], 
                                   skip_paused=True, df=True, fq='pre')
            if len(df) > 0:
                g.etf_info[etf].append(df.close[-1])
                # 保持历史数据长度
                if len(g.etf_info[etf]) > MAX_HISTORY_LENGTH:
                    g.etf_info[etf].pop(0)
        except Exception as e:
            log.error(f"更新ETF {etf} 数据失败: {e}")


def _check_daily_stop_loss(etf, current_price, yesterday_price):
    """检查单日止损"""
    if yesterday_price <= 0:
        return False
    daily_return = current_price / yesterday_price
    return daily_return <= g.daily_stop_loss


def _check_drawdown_stop_loss(etf, current_price):
    """检查回撤止损"""
    if etf not in g.etf_high_price or g.etf_high_price[etf] <= 0:
        return False
    drawdown_ratio = current_price / g.etf_high_price[etf]
    return drawdown_ratio <= g.drawdown_stop_loss


def _update_etf_high_price(etf, current_price):
    """更新ETF最高价"""
    if etf not in g.etf_high_price or current_price > g.etf_high_price[etf]:
        g.etf_high_price[etf] = current_price


def _clear_etf_high_price(etf):
    """清除ETF最高价记录"""
    if etf in g.etf_high_price:
        del g.etf_high_price[etf]


def evaluate_etf_worth(context, etf):
    """
    评估ETF是否值得持有/买入
    
    Args:
        context: 上下文对象
        etf: ETF代码
    
    Returns:
        bool: True=值得持有/买入，False=不值得
    """
    # 数据验证
    if etf is None or etf not in g.etf_info:
        return False
    
    prices = g.etf_info[etf]
    if len(prices) < 2:
        return False
    
    current_price = prices[-1]
    yesterday_price = prices[-2]
    
    # 自动判断是否已持有
    is_holding = etf in context.portfolio.positions
    
    # 已持有：检查止损条件
    if is_holding:
        # 单日止损
        if _check_daily_stop_loss(etf, current_price, yesterday_price):
            log.info(f"{etf} 触发单日止损: {current_price/yesterday_price:.4f}")
            return False
        
        # 回撤止损
        if _check_drawdown_stop_loss(etf, current_price):
            drawdown = (1 - current_price / g.etf_high_price[etf]) * 100
            log.info(f"{etf} 触发回撤止损: 回撤{drawdown:.2f}%")
            return False
        
        # 更新最高价
        _update_etf_high_price(etf, current_price)
        return True
    
    # 未持有：可以买入
    return True


def market_timing_signal(context):
    """
    市场择时信号判断
    
    Returns:
        bool: True=可以持仓, False=建议空仓
    """
    if not g.market_timing:
        return True
    
    try:
        df = attribute_history(MARKET_INDEX, g.market_ma_days + 5, '1d', 
                              ['close'], skip_paused=True, df=True)
        if len(df) < g.market_ma_days:
            return True
        
        current_price = df['close'].iloc[-1]
        ma = df['close'].rolling(window=g.market_ma_days).mean().iloc[-1]
        yesterday_price = df['close'].iloc[-2]
        daily_change = (current_price - yesterday_price) / yesterday_price
        
        # 市场大跌或跌破均线，建议空仓
        if daily_change < g.market_threshold or current_price < ma * 0.98:
            log.info(f"市场择时: 建议谨慎，日涨跌{daily_change*100:.2f}%, "
                    f"价格{current_price:.2f}, 均线{ma:.2f}")
            return False
        
        return True
    except Exception as e:
        log.error(f"市场择时判断失败: {e}")
        return True


def _calculate_momentum_score(etf):
    """
    计算ETF动量得分
    
    Returns:
        float: 动量得分，None表示计算失败
    """
    if len(g.etf_info[etf]) < g.m_days:
        return None
    
    prices = g.etf_info[etf][-g.m_days:]
    if len(prices) < g.m_days:
        return None
    
    try:
        # 对数收益率
        log_returns = np.log(prices)
        x = np.arange(len(log_returns))
        
        # 线性回归
        slope, intercept = np.polyfit(x, log_returns, 1)
        
        # 年化收益率
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        
        # R²（判定系数，衡量趋势稳定性）
        y_pred = slope * x + intercept
        ss_res = np.sum((log_returns - y_pred) ** 2)
        ss_tot = np.sum((log_returns - np.mean(log_returns)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 动量得分 = 年化收益率 × R²
        return annualized_returns * r_squared
    except Exception as e:
        log.error(f"计算ETF {etf} 动量失败: {e}")
        return None


def get_etf_rank(etf_pool):
    """
    计算ETF动量排名
    
    Returns:
        list: 按动量得分排序的ETF列表（从高到低）
    """
    etf_scores = []
    
    for etf in etf_pool:
        score = _calculate_momentum_score(etf)
        if score is not None:
            etf_scores.append((etf, score))
    
    if not etf_scores:
        return []
    
    # 按得分排序
    etf_scores.sort(key=lambda x: x[1], reverse=True)
    rank_list = [etf for etf, _ in etf_scores]
    
    # 记录排名
    df = pd.DataFrame(etf_scores, columns=['etf', 'score'])
    log.info(f"ETF动量排名:\n{df}")
    
    return rank_list


def _sell_etf(etf, reason=""):
    """卖出ETF"""
    order_target_value(etf, 0)
    log.info(f"卖出 {etf}" + (f": {reason}" if reason else ""))
    _clear_etf_high_price(etf)


def _buy_etf(context, etf):
    """买入ETF"""
    total_value = context.portfolio.total_value
    target_value = total_value * g.position_ratio
    
    order_target_value(etf, target_value)
    log.info(f"买入 {etf}, 目标仓位: {target_value:.2f} ({g.position_ratio*100:.0f}%)")
    
    # 初始化最高价
    if etf in g.etf_info and len(g.etf_info[etf]) > 0:
        g.etf_high_price[etf] = g.etf_info[etf][-1]


def trade(context):
    """主交易函数"""
    # 1. 检查市场择时信号
    if not market_timing_signal(context):
        hold_etf = list(context.portfolio.positions)
        if hold_etf:
            _sell_etf(hold_etf[0], "市场择时信号")
        return
    
    # 2. 获取ETF动量排名
    rank_list = get_etf_rank(g.etf_pool)
    if not rank_list:
        log.warning("无法获取ETF排名，跳过交易")
        return
    
    selected_etf = rank_list[0]
    hold_list = list(context.portfolio.positions)
    hold_etf = hold_list[0] if hold_list else None
    
    # 3. 评估ETF价值
    selected_worth = evaluate_etf_worth(context, selected_etf)
    hold_worth = evaluate_etf_worth(context, hold_etf) if hold_etf else False
    
    log.info(f"持有: {hold_etf} (价值={hold_worth}), "
            f"选择: {selected_etf} (价值={selected_worth})")
    
    # 4. 卖出逻辑
    if hold_etf:
        g.etf_pre = hold_etf
        
        # 调仓：选择的ETF与持有的不同
        if selected_etf != hold_etf:
            _sell_etf(hold_etf, "调仓")
        # 止损：触发止损条件
        elif not hold_worth:
            _sell_etf(hold_etf, "止损")
        # 继续持有：更新最高价
        else:
            log.info(f"继续持有: {hold_etf}")
            if hold_etf in g.etf_info and len(g.etf_info[hold_etf]) > 0:
                _update_etf_high_price(hold_etf, g.etf_info[hold_etf][-1])
    
    # 5. 买入逻辑
    if selected_worth:
        # 调仓买入：选择的ETF与之前不同
        if selected_etf != g.etf_pre:
            _buy_etf(context, selected_etf)
        # 恢复买入：之前止损卖出，现在可以重新买入
        elif not hold_list and selected_etf == g.etf_pre:
            _buy_etf(context, selected_etf)
