from jqdata import *

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    g.stocks = ['000001.XSHE', '600000.XSHG']
    run_daily(trade, time='every_bar')

def trade(context):
    order(g.stocks[0], 100)
    order(g.stocks[1], 200)