"""
策略测试运行器

自动发现并测试 tests/strategies/ 目录下的所有策略文件
配置文件：tests/strategies/config.yaml
"""
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pytest
import yaml

# 加载 .env 环境变量
from bullet_trade.utils.env_loader import load_env

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 确保 jqdata 指向项目内置兼容模块，避免被系统同名包覆盖
_local_jq_path = (project_root / "jqdata.py").resolve()
try:
    import jqdata as _jq_module  # type: ignore
except ImportError:
    _jq_module = None  # type: ignore

if not _jq_module or Path(getattr(_jq_module, "__file__", "")).resolve() != _local_jq_path:
    spec = importlib.util.spec_from_file_location("jqdata", _local_jq_path)
    _jq_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(_jq_module)  # type: ignore[arg-type]
sys.modules["jqdata"] = _jq_module  # type: ignore[arg-type]

from bullet_trade.core.engine import BacktestEngine


def load_config() -> Dict[str, Any]:
    """
    加载策略配置文件
    
    Returns:
        Dict[str, Any]: 配置字典
    """
    config_file = Path(__file__).parent / 'strategies' / 'config.yaml'
    
    if not config_file.exists():
        print(f"警告: 配置文件 {config_file} 不存在，使用默认配置")
        return {'default': get_default_config()}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config if config else {'default': get_default_config()}
    except Exception as e:
        print(f"警告: 加载配置文件失败: {e}，使用默认配置")
        return {'default': get_default_config()}


def discover_strategies() -> List[Tuple[str, Path]]:
    """
    发现所有策略文件
    
    Returns:
        List[Tuple[str, Path]]: 策略名称和文件路径的列表
    """
    strategies_dir = Path(__file__).parent / 'strategies'
    strategy_files = []
    
    if strategies_dir.exists():
        for file_path in strategies_dir.glob('*.py'):
            # 跳过 __init__.py、测试脚本
            if file_path.name.startswith('__') or file_path.name.startswith('test_'):
                continue
            
            strategy_name = file_path.stem
            strategy_files.append((strategy_name, file_path))
    
    return sorted(strategy_files)


def load_strategy_module(strategy_path: Path):
    """
    动态加载策略模块
    
    Args:
        strategy_path: 策略文件路径
        
    Returns:
        module: 加载的策略模块
    """
    spec = importlib.util.spec_from_file_location(
        f"strategy_{strategy_path.stem}", 
        strategy_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_default_config() -> Dict[str, Any]:
    """
    获取默认的策略测试配置
    
    Returns:
        Dict[str, Any]: 默认配置
    """
    return {
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'capital_base': 100000,
        'frequency': 'daily',
        'benchmark': '000300.XSHG',
        'expected': {}
    }


def get_strategy_config(strategy_name: str, all_configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    获取特定策略的配置
    
    Args:
        strategy_name: 策略名称
        all_configs: 所有配置
        
    Returns:
        Dict[str, Any]: 策略配置
    """
    # 优先使用策略特定配置
    if strategy_name in all_configs:
        config = all_configs['default'].copy() if 'default' in all_configs else get_default_config()
        config.update(all_configs[strategy_name])
        return config
    
    # 使用默认配置
    if 'default' in all_configs:
        return all_configs['default'].copy()
    
    return get_default_config()


def validate_results(results: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    """
    验证回测结果是否符合预期
    
    Args:
        results: 回测结果
        expected: 期望的结果约束
        
    Returns:
        List[str]: 验证失败的错误信息列表
    """
    errors = []
    
    for metric, constraints in expected.items():
        if metric not in results:
            continue
            
        actual_value = results[metric]
        
        # 检查最小值约束
        if 'min' in constraints:
            min_value = constraints['min']
            if actual_value < min_value:
                errors.append(
                    f"{metric} = {actual_value:.4f} 小于期望最小值 {min_value:.4f}"
                )
        
        # 检查最大值约束
        if 'max' in constraints:
            max_value = constraints['max']
            if actual_value > max_value:
                errors.append(
                    f"{metric} = {actual_value:.4f} 大于期望最大值 {max_value:.4f}"
                )
    
    return errors


# 加载配置
ALL_CONFIGS = load_config()

# 发现所有策略
STRATEGIES = discover_strategies()


@pytest.mark.parametrize("strategy_name,strategy_path", STRATEGIES)
def test_strategy(strategy_name: str, strategy_path: Path):
    """
    测试单个策略
    
    Args:
        strategy_name: 策略名称
        strategy_path: 策略文件路径
    """
    print(f"\n{'='*60}")
    print(f"测试策略: {strategy_name}")
    print(f"文件路径: {strategy_path}")
    print(f"{'='*60}")
    
    # 特定策略前置检查
    if strategy_name == "data_api_temporal_guards":
        load_env()
        if not os.getenv("JQDATA_USERNAME") or not os.getenv("JQDATA_PASSWORD"):
            pytest.skip("缺少 JQDATA_USERNAME/JQDATA_PASSWORD，跳过 data_api_temporal_guards")
        try:
            import jqdatasdk  # noqa: F401
        except Exception:
            pytest.skip("未安装 jqdatasdk，跳过 data_api_temporal_guards")
    if strategy_name == "strategy_small_cap_direct_provider_access":
        load_env()
        if not os.getenv("JQDATA_USERNAME") or not os.getenv("JQDATA_PASSWORD"):
            pytest.skip("缺少 JQDATA_USERNAME/JQDATA_PASSWORD，跳过直连示例")
        try:
            import jqdatasdk  # noqa: F401
        except Exception:
            pytest.skip("未安装 jqdatasdk，跳过直连示例")
        try:
            from xtquant import xtdata  # noqa: F401
        except Exception:
            pytest.skip("缺少 miniQMT/xtquant 环境，跳过直连示例")

    # 加载策略模块前确保 jqdata 仍指向本地兼容模块（防止被其他测试污染）
    _local_jq_path = (project_root / "jqdata.py").resolve()
    jq_module = sys.modules.get("jqdata")
    if Path(getattr(jq_module, "__file__", "")).resolve() != _local_jq_path:
        spec = importlib.util.spec_from_file_location("jqdata", _local_jq_path)
        jq_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(jq_module)  # type: ignore[arg-type]
        sys.modules["jqdata"] = jq_module  # type: ignore[arg-type]

    # 加载策略模块
    strategy_module = load_strategy_module(strategy_path)
    
    # 从配置文件获取策略配置（不再从策略文件读取）
    config = get_strategy_config(strategy_name, ALL_CONFIGS)
    
    print(f"\n策略配置:")
    print(f"  回测期间: {config['start_date']} ~ {config['end_date']}")
    print(f"  初始资金: {config['capital_base']:,.0f}")
    print(f"  运行频率: {config['frequency']}")
    print(f"  基准指数: {config['benchmark']}")
    
    # 检查必需的策略函数
    assert hasattr(strategy_module, 'initialize'), \
        f"策略 {strategy_name} 缺少 initialize 函数"
    
    # 创建回测引擎
    engine = BacktestEngine(
        initialize=strategy_module.initialize,
        handle_data=getattr(strategy_module, 'handle_data', None),
        process_initialize=getattr(strategy_module, 'process_initialize', None),
        after_trading_end=getattr(strategy_module, 'after_trading_end', None),
        before_trading_start=getattr(strategy_module, 'before_trading_start', None),
    )
    
    # 运行回测
    try:
        results = engine.run(
            start_date=config['start_date'],
            end_date=config['end_date'],
            capital_base=config['capital_base'],
            frequency=config['frequency'],
            benchmark=config['benchmark']
        )
        
        # 打印关键指标
        print(f"\n回测结果:")
        print(f"  总收益率: {results.get('total_returns', 0):.2%}")
        print(f"  年化收益率: {results.get('annual_returns', 0):.2%}")
        print(f"  基准收益率: {results.get('benchmark_returns', 0):.2%}")
        print(f"  阿尔法: {results.get('alpha', 0):.4f}")
        print(f"  贝塔: {results.get('beta', 0):.4f}")
        print(f"  夏普比率: {results.get('sharpe', 0):.4f}")
        print(f"  最大回撤: {results.get('max_drawdown', 0):.2%}")
        print(f"  胜率: {results.get('win_rate', 0):.2%}")
        
        # 验证结果
        if config.get('expected'):
            print(f"\n验证结果约束...")
            errors = validate_results(results, config['expected'])
            
            if errors:
                error_msg = "\n".join(errors)
                pytest.fail(f"策略 {strategy_name} 未满足预期约束:\n{error_msg}")
            else:
                print(f"  ✓ 所有约束条件均已满足")
        
        print(f"\n{'='*60}")
        print(f"策略 {strategy_name} 测试通过 ✓")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n策略运行失败: {str(e)}")
        raise


def test_no_strategies_warning():
    """
    如果没有发现任何策略，给出警告
    """
    if not STRATEGIES:
        pytest.skip("未发现任何策略文件，请在 tests/strategies/ 目录下添加策略文件")


if __name__ == '__main__':
    # 直接运行此文件时，使用 pytest
    pytest.main([__file__, '-v', '-s'])
