# 交易支撑：聚宽模拟盘远程实盘

聚焦聚宽 Research/模拟盘如何调用远程 QMT server 完成真实下单，含截图与常见问题。

## 步骤总览
1) 启动远程 QMT server（数据 + 交易）并放行防火墙。  
2) 上传 helper 到聚宽根目录。  
3) 在策略里配置 host/port/token，调用 helper 下单/查询。

## 1. 启动 QMT server
```bash
bullet-trade server --listen 0.0.0.0 --port 58620 --token secret ^
  --enable-data --enable-broker ^
  --accounts main=123456:stock ^
  --data-path "C:\国金QMT交易端\userdata_mini"
```
防火墙放行提示：
![server-firewall](assets/server-firewall.png)

服务端日志示例：
![joinquant-server-qmt](assets/joinquant-server-qmt.png)

## 2. 聚宽 Research 上传 helper
- 位置：聚宽根目录 `/`
- 文件：`helpers/bullet_trade_jq_remote_helper.py`
- 操作截图：
![joinquant-sever-reserch](assets/joinquant-sever-reserch.png)

## 3. 在策略里调用
```python
import bullet_trade_jq_remote_helper as bt

bt.configure(
    host="your.server.ip",
    port=58620,
    token="secret",
    account_key="main",    # 对应 --accounts 别名
    sub_account_id=None,   # 可选
)
bt.bind_clients_to_globals()  # 将数据/券商客户端挂到 g

def initialize(context):
    set_benchmark('000300.XSHG')

def handle_data(context, data):
    bt.order('000001.XSHE', 100)          # 买入
    bt.order_value('000002.XSHE', 50000)  # 按金额买入
    positions = bt.get_positions()
```
支持接口：`order/order_value/order_target/order_target_value`、`cancel_order/get_order_status/get_open_orders/get_orders/get_trades`、`get_account/get_positions`。

### 运行效果
聚宽策略日志+持仓：
![joinquant-remote-live-support](assets/joinquant-remote-live-support.png)

同步/异步下单时间线（默认同步等待 16s，示例 0.5s 成交）：
![sync-order](assets/sync-order.png)

## 常见问题
- 401 未授权：token 不一致或被 `--allowlist` 限制。
- 超时/未成交：查看服务端日志，按需调整 `wait_timeout` 或改用异步；确认行情正常。
- 无行情：服务端需启用 `--enable-data` 且 QMT 已登录。
- 多账户：server `--accounts a=...`，策略里 `account_key='a'` 指定。

更多 server 侧参数与安全提示见 [QMT server](qmt-server.md)；配置变量详见 [配置总览](config.md)。
