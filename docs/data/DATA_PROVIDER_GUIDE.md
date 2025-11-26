# BulletTrade 数据提供者使用指南

本目录包含使用 BulletTrade 不同数据提供者的示例 notebook。


## 🆚 数据提供者对比

### JQData Provider 

**优点：**
- ✅ 数据全面：股票、基金、期货、期权等
- ✅ 历史悠久：可获取多年历史数据
- ✅ 自动缓存：BulletTrade 自动缓存到本地
- ✅ 稳定可靠：聚宽官方维护

**缺点：**
- ⚠️ 需要网络：必须连接到聚宽服务器
- ⚠️ 可能有延迟：网络延迟影响速度
- ⚠️ 账号限制：需要聚宽账号，可能有调用频率限制

**适用场景：**
- 📊 回测历史策略
- 🔍 数据研究与分析
- 📚 学习量化交易

### MiniQMT Provider 

**优点：**
- ✅ 本地数据：无需网络，速度极快
- ✅ 实盘对接：可直接连接 QMT 实盘交易
- ✅ 数据安全：数据不出本地
- ✅ 格式兼容：同时支持 QMT 和聚宽代码格式

**缺点：**
- ⚠️ 数据范围受限：只有本地 QMT 已下载的数据
- ⚠️ 需要安装：必须先安装 miniQMT/xtquant
- ⚠️ 配置复杂：需要正确配置数据目录

**适用场景：**
- 🚀 实盘交易（结合 QMT）
- ⚡ 需要极速数据访问
- 🔒 对数据安全有要求
- 💻 已有 QMT 环境

## 🔧 配置说明

### 1. JQData 配置（.env 示例）

```env
# 默认数据源设置为 jqdata
DEFAULT_DATA_PROVIDER=jqdata

# 可选：通用缓存目录（会自动创建 jqdatasdk 等子目录）
#DATA_CACHE_DIR=c:\\bt_cache

# JQData 认证信息
JQDATA_USERNAME=your_username
JQDATA_PASSWORD=your_password
```

### 2. MiniQMT 配置（.env 示例）

```env
# 默认数据源设置为 qmt
DEFAULT_DATA_PROVIDER=qmt

# MiniQMT 数据目录（必需）
QMT_DATA_PATH=C:\国金QMT交易端模拟\userdata_mini

# 是否自动下载数据
MINIQMT_AUTO_DOWNLOAD=true

# 交易日市场代码
MINIQMT_MARKET=SH
```

## 📝 代码示例

### 使用 JQData Provider

```python
from bullet_trade.data.api import get_price, set_data_provider

# 设置使用 jqdata
set_data_provider('jqdata')

# 获取日线数据（使用聚宽格式代码）
df = get_price('601318.XSHG', '2025-07-01', '2025-07-31', fq=None)

# 获取分钟数据
df_1m = get_price('601318.XSHG', '2025-07-01 09:25:00', '2025-07-01 09:35:00', 
                  frequency='1m', fq=None)
```

### 使用 MiniQMT Provider

```python
from bullet_trade.data.api import get_price, set_data_provider

# 设置使用 qmt
set_data_provider('qmt')

# 获取日线数据（支持 QMT 格式和聚宽格式）
df = get_price('601318.SH', '2025-07-01', '2025-07-31', fq=None)
# 或
df = get_price('601318.XSHG', '2025-07-01', '2025-07-31', fq=None)

# 获取分钟数据
df_1m = get_price('601318.SH', '2025-07-01 09:25:00', '2025-07-01 09:35:00', 
                  frequency='1m', fq=None)
```

## 🔄 切换数据源

在运行时可以随时切换数据源：

```python
from bullet_trade.data.api import set_data_provider

# 切换到 JQData
set_data_provider('jqdata')

# 切换到 MiniQMT
set_data_provider('qmt')

# 切换到 Tushare（如果配置了）
set_data_provider('tushare')
```

## 🎯 代码格式对照表

| 交易所 | 聚宽格式（JQData） | QMT 格式（MiniQMT） | 说明 |
|--------|-------------------|-------------------|------|
| 上海 | `601318.XSHG` | `601318.SH` | MiniQMT 两种都支持 |
| 深圳 | `000001.XSHE` | `000001.SZ` | MiniQMT 两种都支持 |

**注意：** 
- JQData Provider **只支持聚宽格式**（`.XSHG`/`.XSHE`）
- MiniQMT Provider **两种格式都支持**，自动转换

## 💡 最佳实践

### 开发阶段
- 使用 **JQData Provider** 进行策略开发和回测
- 数据全面，便于研究和验证

### 实盘阶段
- 使用 **MiniQMT Provider** 进行实盘交易
- 本地数据，速度快，延迟低

### 统一代码
- 建议在策略中使用**聚宽格式代码**（`.XSHG`/`.XSHE`）
- 这样切换数据源时无需修改代码
- MiniQMT Provider 会自动转换

## 🐛 常见问题

### Q1: 如何知道当前使用的是哪个数据源？

```python
from bullet_trade.data.api import get_data_provider

provider = get_data_provider()
print(f"当前数据源: {provider.name}")
```

### Q2: JQData 认证失败怎么办？

检查 `.env` 文件中的配置：
- `JQDATA_USERNAME` 是否正确（手机号）
- `JQDATA_PASSWORD` 是否正确


### Q3: MiniQMT 找不到数据目录？

确认配置：
```env
QMT_DATA_PATH=C:\国金QMT交易端模拟\userdata_mini
```
- 路径是否存在
- 路径是否正确（根据实际安装目录调整）
- QMT 是否已经下载了相应的数据

### Q4: 数据格式不一致怎么办？

- **推荐**：在策略中统一使用聚宽格式（`.XSHG`/`.XSHE`）
- MiniQMT Provider 会自动转换格式
- 这样切换数据源时代码不需要修改

## 相关文档

- [聚宽数据](DATA_PROVIDER_JQDATA.md)
- [MiniQMT 数据](DATA_PROVIDER_MINIQMT.md)
- [Tushare 数据](DATA_PROVIDER_TUSHARE.md)
