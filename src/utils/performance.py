import time
from typing import Dict, Any, Optional
from functools import wraps
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TimingStats:
    """用于记录和管理执行时间统计的工具类"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据"""
        self.stats = {}
        self.current_operation = None
        self.start_time = None

    def start_timer(self, operation: str):
        """开始计时特定操作"""
        self.current_operation = operation
        self.start_time = time.time()
        return self

    def stop_timer(self) -> float:
        """停止当前操作的计时并记录"""
        if self.current_operation is None or self.start_time is None:
            return 0.0

        elapsed = time.time() - self.start_time

        if self.current_operation not in self.stats:
            self.stats[self.current_operation] = {
                'count': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }

        stats = self.stats[self.current_operation]
        stats['count'] += 1
        stats['total_time'] += elapsed
        stats['min_time'] = min(stats['min_time'], elapsed)
        stats['max_time'] = max(stats['max_time'], elapsed)

        self.current_operation = None
        self.start_time = None

        return elapsed

    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计数据的副本"""
        result = {}

        for op, stats in self.stats.items():
            result[op] = stats.copy()
            if stats['count'] > 0:
                result[op]['avg_time'] = stats['total_time'] / stats['count']
            else:
                result[op]['avg_time'] = 0

        return result

    def get_summary(self) -> Dict[str, float]:
        """获取简化的统计摘要，只包含总时间"""
        return {op: op_stats['total_time'] for op, op_stats in self.stats.items()}

    def log_stats(self, level=logging.DEBUG):
        """将当前统计数据记录到日志"""
        stats = self.get_stats()
        for op, op_stats in stats.items():
            logger.log(level, f"{op}: count={op_stats['count']}, "
                              f"total={op_stats['total_time']:.4f}s, "
                              f"avg={op_stats.get('avg_time', 0):.4f}s, "
                              f"min={op_stats['min_time']:.4f}s, "
                              f"max={op_stats['max_time']:.4f}s")

    @contextmanager
    def measure(self, operation: str):
        """
        上下文管理器方法，用于在with语句中测量操作时间

        用法:
            with timing_stats.measure("操作名称"):
                # 要测量的代码
        """
        self.start_timer(operation)
        try:
            yield
        finally:
            self.stop_timer()


def timed(operation: Optional[str] = None, stats_instance: Optional[TimingStats] = None):
    """
    计时装饰器，用于自动记录函数执行时间

    用法：
        timing_stats = TimingStats()

        @timed("数据库查询", timing_stats)
        def query_database(query):
            # 函数体

        # 或者自动使用函数名
        @timed(stats_instance=timing_stats)
        def process_data():
            # 函数体
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 如果没有提供统计实例，只记录日志
            nonlocal operation
            op_name = operation or func.__name__

            start = time.time()

            if stats_instance is not None:
                stats_instance.start_timer(op_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start

                if stats_instance is not None:
                    stats_instance.stop_timer()

                logger.debug(f"{op_name} 耗时: {elapsed:.4f}秒")

        return wrapper

    return decorator


# 创建全局统计实例，可以在任何地方导入使用
global_timing_stats = TimingStats()