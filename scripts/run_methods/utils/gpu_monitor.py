#!/usr/bin/env python
"""
GPU监控工具模块

提供后台GPU监控功能，用于记录训练过程中的GPU显存使用和利用率。

使用方法:
    from utils.gpu_monitor import GPUMonitor, save_gpu_stats
    
    # 启动监控
    monitor = GPUMonitor(gpu_id=0, sampling_interval=1.0)
    monitor.start()
    
    try:
        # 训练代码...
        pass
    finally:
        # 停止监控并保存统计数据
        stats = monitor.stop()
        save_gpu_stats(stats, output_dir, method_name='scmiac', gpu_id=0)
"""

import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn(
        "pynvml not installed. GPU monitoring will be disabled. "
        "Install it with: pip install pynvml"
    )


class GPUMonitor:
    """后台GPU监控线程
    
    在单独的线程中定期采样GPU显存使用和利用率，记录峰值和平均值。
    
    Args:
        gpu_id: GPU设备ID（默认0）
        sampling_interval: 采样间隔（秒，默认1.0）
    """
    
    def __init__(self, gpu_id: int = 0, sampling_interval: float = 1.0):
        if not PYNVML_AVAILABLE:
            raise RuntimeError(
                "pynvml is required for GPU monitoring. "
                "Install it with: pip install pynvml"
            )
        
        self.gpu_id = gpu_id
        self.sampling_interval = sampling_interval
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        
        # 监控数据
        self.peak_memory_mb = 0.0
        self.utilization_samples = []
        self.memory_samples = []
        
        # NVML初始化标志
        self._nvml_initialized = False
    
    def start(self):
        """启动GPU监控线程"""
        if not PYNVML_AVAILABLE:
            print("Warning: pynvml not available, GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            
            # 验证GPU ID是否有效
            device_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_id >= device_count:
                raise ValueError(
                    f"Invalid GPU ID {self.gpu_id}. "
                    f"Available GPUs: 0-{device_count - 1}"
                )
            
            # 获取GPU名称
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            print(f"GPU monitoring started for GPU {self.gpu_id}: {gpu_name}")
            print(f"Sampling interval: {self.sampling_interval}s")
            
            self.is_running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"Warning: Failed to start GPU monitoring: {e}")
            if self._nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                self._nvml_initialized = False
    
    def stop(self) -> Dict:
        """停止GPU监控并返回统计数据
        
        Returns:
            包含GPU统计信息的字典
        """
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
            self._nvml_initialized = False
        
        return self.get_stats()
    
    def _monitor_loop(self):
        """GPU监控循环（在后台线程中运行）"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
            
            while self.is_running:
                try:
                    # 获取显存使用
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_mb = mem_info.used / (1024 ** 2)
                    
                    # 获取GPU利用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                    
                    # 记录数据
                    self.memory_samples.append(memory_mb)
                    self.utilization_samples.append(gpu_util)
                    self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                    
                except Exception as e:
                    # 忽略单次采样错误，继续监控
                    print(f"Warning: GPU sampling error: {e}")
                
                time.sleep(self.sampling_interval)
        
        except Exception as e:
            print(f"Error in GPU monitoring loop: {e}")
    
    def get_stats(self) -> Dict:
        """获取GPU统计数据
        
        Returns:
            包含以下字段的字典：
            - peak_memory_mb: 峰值显存（MB）
            - peak_memory_gb: 峰值显存（GB）
            - avg_utilization: 平均GPU利用率（%）
            - max_utilization: 最大GPU利用率（%）
            - sampling_count: 采样次数
        """
        if not self.utilization_samples:
            return {
                'peak_memory_mb': 0.0,
                'peak_memory_gb': 0.0,
                'avg_utilization': 0.0,
                'max_utilization': 0.0,
                'sampling_count': 0
            }
        
        avg_util = sum(self.utilization_samples) / len(self.utilization_samples)
        max_util = max(self.utilization_samples)
        
        return {
            'peak_memory_mb': self.peak_memory_mb,
            'peak_memory_gb': self.peak_memory_mb / 1024,
            'avg_utilization': avg_util,
            'max_utilization': max_util,
            'sampling_count': len(self.utilization_samples)
        }


def save_gpu_stats(
    stats: Dict,
    output_dir: Path,
    method_name: str,
    gpu_id: int = 0,
    csv_path: Optional[Path] = None
):
    """保存GPU统计数据到CSV文件
    
    数据会保存到指定的CSV文件中。
    如果文件已存在且包含该方法的旧记录，会先删除旧记录再追加新记录。
    
    Args:
        stats: GPU统计数据字典（由 GPUMonitor.get_stats() 返回）
        output_dir: 输出目录（当csv_path为None时使用，默认保存为output_dir/gpu.csv）
        method_name: 方法名称（如 'scmiac', 'scanvi' 等）
        gpu_id: GPU设备ID
        csv_path: 自定义CSV文件路径（可选，如果提供则忽略output_dir）
    """
    if csv_path is not None:
        gpu_csv_file = Path(csv_path)
        gpu_csv_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_csv_file = output_dir / "gpu.csv"
    
    # 如果文件不存在，创建表头
    if not gpu_csv_file.exists():
        with open(gpu_csv_file, 'w') as f:
            f.write("method,gpu_id,peak_memory_mb,peak_memory_gb,"
                   "avg_utilization_percent,max_utilization_percent,"
                   "sampling_count,timestamp\n")
        print(f"Created GPU stats file: {gpu_csv_file}")
    
    # 读取现有数据，删除该方法的旧记录
    try:
        lines = []
        with open(gpu_csv_file, 'r') as f:
            lines = f.readlines()
        
        # 保留表头和其他方法的记录
        filtered_lines = [lines[0]]  # 表头
        for line in lines[1:]:
            if not line.startswith(f"{method_name},"):
                filtered_lines.append(line)
        
        # 如果有变化，重写文件
        if len(filtered_lines) != len(lines):
            with open(gpu_csv_file, 'w') as f:
                f.writelines(filtered_lines)
    
    except Exception as e:
        print(f"Warning: Failed to remove old GPU stats: {e}")
    
    # 追加新记录
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(gpu_csv_file, 'a') as f:
        f.write(
            f"{method_name},{gpu_id},"
            f"{stats['peak_memory_mb']:.2f},{stats['peak_memory_gb']:.3f},"
            f"{stats['avg_utilization']:.2f},{stats['max_utilization']:.2f},"
            f"{stats['sampling_count']},{timestamp}\n"
        )
    
    print(f"GPU stats saved to: {gpu_csv_file}")
    print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB "
          f"({stats['peak_memory_gb']:.3f} GB)")
    print(f"  Avg utilization: {stats['avg_utilization']:.2f}%")
    print(f"  Max utilization: {stats['max_utilization']:.2f}%")
    print(f"  Samples: {stats['sampling_count']}")
