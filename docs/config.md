# 配置总览（回测/实盘/远程）

统一列出常用环境变量，避免在多页面重复。按照场景分组，可结合 `.env` 模板使用。

## 通用
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `DEFAULT_DATA_PROVIDER` | 否 | `jqdata` | 默认行情源，回测/实盘共用（`jqdata`/`tushare`/`qmt`/`local`） |
| `DEFAULT_BROKER` | 否 | `qmt` | 默认券商/交易通道（`simulator`/`qmt`/`qmt-remote`） |
| `DATA_CACHE_DIR` | 否 | `~/.bullet-trade/cache` | 行情缓存根目录，子目录按数据源名自动创建；留空禁用缓存 |
| `LOG_DIR` | 否 | `logs` | 日志目录 |
| `LOG_LEVEL` | 否 | `INFO` | 控制台日志级别（`DEBUG`/`INFO`/`WARNING`/`ERROR`） |
| `LOG_FILE_LEVEL` | 否 | 跟随 `LOG_LEVEL` | 文件日志级别，未设置则与 `LOG_LEVEL` 相同 |
| `RUNTIME_DIR` | 否 | `runtime` | 运行态/持久化目录（含 g.pkl、live_state.json） |

## 回测
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `DEFAULT_DATA_PROVIDER` | 是 | `jqdata` | 回测行情源（`jqdata`/`tushare`/`qmt`/`local`） |
| `JQDATA_USERNAME`/`JQDATA_PASSWORD` | 视数据源 | `your_user`/`your_pwd` | 聚宽数据账号 |
| `TUSHARE_TOKEN` | 选 | `your_token` | 需 `tushare` 时配置 |
| `MINIQMT_*` | 否 | 见实盘 | 仅当用 `qmt` 做行情源 |
| `LOCAL_DB_PATH` | 选 | `/path/to/qmt_data.db` | 使用 `local` 数据源时指定 SQLite 数据库路径，未设置则使用默认路径 `data/qmt_data.db` |

## 本地实盘（QMT/模拟）
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `DEFAULT_BROKER` | 是 | `qmt` 或 `simulator` | 交易通道 |
| `QMT_DATA_PATH` | 是 | `C:\国金QMT交易端\userdata_mini` | xtquant 数据目录 |
| `QMT_ACCOUNT_ID` | 是 | `123456` | QMT 账户 |
| `QMT_ACCOUNT_TYPE` | 否 | `stock` | 账户类型 |
| `MINIQMT_AUTO_DOWNLOAD` | 否 | `true` | 自动下载 MiniQMT |
| `QMT_SESSION_ID` | 否 | `0` | 会话 ID，可选 |

## 风控（实盘）
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `MAX_ORDER_VALUE` | 否 | `100000` | 单笔订单金额上限，超出直接拒单。 |
| `MAX_DAILY_TRADE_VALUE` | 否 | `500000` | 单日累计成交金额上限，超出直接拒单。 |
| `MAX_DAILY_TRADES` | 否 | `100` | 单日最大交易笔数，超出直接拒单。 |
| `MAX_STOCK_COUNT` | 否 | `20` | 最大持仓标的数（仅买入检查）。 |
| `MAX_POSITION_RATIO` | 否 | `20` | 单标下单金额占总资产上限，需要在风控检查时提供账户总资产才会生效。 |
| `STOP_LOSS_RATIO` | 否 | `5` | 止损阈值，供 `check_stop_loss` 辅助判断，需策略自行下撤单。 |
| `RISK_CHECK_ENABLED` | 否 | `false` | 后台风控巡检开关，间隔由 `RISK_CHECK_INTERVAL` 控制（默认 `300` 秒）。 |

## 远程实盘（qmt-remote）
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `DEFAULT_BROKER` | 是 | `qmt-remote` | 远程通道 |
| `QMT_SERVER_HOST`/`QMT_SERVER_PORT` | 是 | `10.0.0.8`/`58620` | 远程服务地址 |
| `QMT_SERVER_TOKEN` | 是 | `secret` | 访问令牌 |
| `QMT_SERVER_ACCOUNT_KEY` | 否 | `main` | 多账户时指定 |

## 聚宽模拟盘接入远程实盘
| 变量 | 必填 | 示例/默认 | 作用 |
| --- | --- | --- | --- |
| `BT_REMOTE_HOST`/`BT_REMOTE_PORT` | 是 | `10.0.0.8`/`58620` | 远程 QMT server |
| `BT_REMOTE_TOKEN` | 是 | `secret` | 访问令牌 |
| `ACCOUNT_KEY`/`SUB_ACCOUNT` | 视账户 | `main`/`子账户` | 选填，匹配远程账户 |

> 提示：敏感信息不要入库，按需覆盖到本地 `.env`。***
