from jqdata import *

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    g.stocks = ['159629.XSHE']  # 修改为159629
    run_daily(trade, time='02:55')  # 改为每天2:55执行

def trade(context):
    order(g.stocks[0], 1)  # 买一股
