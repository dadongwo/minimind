"""
MiniMind 设备兼容性工具模块
支持 NVIDIA CUDA 和 AMD ROCm GPU 的统一管理
"""

import torch
import os
import warnings
from typing import Optional, Union, Tuple


class DeviceManager:
    """统一的设备管理器，支持CUDA和ROCm"""
    
    def __init__(self):
        self.device_type = self._detect_device_type()
        self.backend_type = self._get_backend_type()
        
    def _detect_device_type(self) -> str:
        """检测可用的GPU类型"""
        # 首先检查DirectML是否可用（Windows AMD GPU的最佳选择）
        if self._is_directml_available():
            return "directml"

        # 然后检查是否为ROCm环境
        if self._is_rocm_available():
            return "rocm"

        # 最后检测NVIDIA CUDA
        if torch.cuda.is_available():
            # 进一步验证是否真的是NVIDIA GPU
            try:
                # 尝试获取GPU名称来区分NVIDIA和AMD
                if torch.cuda.device_count() > 0:
                    device_name = torch.cuda.get_device_name(0).lower()
                    if 'nvidia' in device_name or 'geforce' in device_name or 'tesla' in device_name or 'quadro' in device_name:
                        return "cuda"
                    elif 'amd' in device_name or 'radeon' in device_name or 'gfx' in device_name:
                        return "rocm"
                return "cuda"  # 默认假设是CUDA
            except (RuntimeError, AttributeError, IndexError):
                return "cuda"

        return "cpu"

    def _is_rocm_available(self) -> bool:
        """检查ROCm是否可用"""
        # 方法1: 检查torch.version.hip
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return True
        except (AttributeError, RuntimeError):
            pass

        # 方法2: 检查环境变量
        rocm_paths = ['ROCM_PATH', 'HIP_PATH', 'ROCM_HOME']
        if any(os.environ.get(path) for path in rocm_paths):
            return True

        # 方法3: 检查PyTorch编译标志
        try:
            if torch.cuda.is_available():
                # 检查是否使用ROCm编译
                device_name = torch.cuda.get_device_name(0).lower()
                if any(keyword in device_name for keyword in ['amd', 'radeon', 'gfx']):
                    return True
        except (RuntimeError, AttributeError, IndexError):
            pass

        return False

    def _is_directml_available(self) -> bool:
        """检查DirectML是否可用"""
        try:
            import torch_directml
            # 尝试创建DirectML设备
            device = torch_directml.device()
            # 测试基础操作
            test_tensor = torch.randn(2, 2, device=device)
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _get_backend_type(self) -> str:
        """获取分布式训练后端类型"""
        if self.device_type == "cuda":
            return "nccl"
        elif self.device_type == "rocm":
            return "nccl"  # ROCm也支持NCCL
        elif self.device_type == "directml":
            return "gloo"  # DirectML使用gloo后端
        else:
            return "gloo"
    
    def is_gpu_available(self) -> bool:
        """检查是否有GPU可用"""
        return self.device_type in ["cuda", "rocm", "directml"]
    
    def get_device_count(self) -> int:
        """获取GPU数量"""
        if self.device_type == "cuda":
            return torch.cuda.device_count()
        elif self.device_type == "rocm":
            return torch.cuda.device_count()  # ROCm使用相同的API
        elif self.device_type == "directml":
            return 1  # DirectML通常只支持一个设备
        else:
            return 0
    
    def get_default_device(self, device_id: int = 0):
        """获取默认设备

        Returns:
            Union[str, torch.device]: 设备标识符或设备对象
            - CUDA/ROCm: 返回字符串格式 "cuda:0"
            - DirectML: 返回torch_directml.device对象
            - CPU: 返回字符串 "cpu"
        """
        if self.is_gpu_available():
            # ROCm使用cuda:x格式，因为PyTorch ROCm使用相同的API
            if self.device_type == "rocm":
                return f"cuda:{device_id}"
            elif self.device_type == "directml":
                # DirectML使用特殊的设备格式
                import torch_directml
                return torch_directml.device(device_id)
            else:
                return f"{self.device_type}:{device_id}"
        else:
            return "cpu"
    
    def set_device(self, device_id: int):
        """设置当前设备"""
        if not self.is_gpu_available():
            warnings.warn(f"尝试设置GPU设备{device_id}，但没有可用的GPU")
            return

        if device_id >= self.get_device_count():
            raise ValueError(f"设备ID {device_id} 超出可用设备数量 {self.get_device_count()}")

        try:
            if self.device_type in ["cuda", "rocm"]:
                torch.cuda.set_device(device_id)  # ROCm使用相同的API
        except Exception as e:
            raise RuntimeError(f"设置设备 {device_id} 失败: {e}")
    
    def manual_seed(self, seed: int):
        """设置GPU随机种子"""
        torch.manual_seed(seed)
        if self.device_type == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        elif self.device_type == "rocm":
            torch.cuda.manual_seed(seed)  # ROCm使用相同的API
            torch.cuda.manual_seed_all(seed)
    
    def get_amp_context(self, enabled: bool = True):
        """获取自动混合精度上下文"""
        if not self.is_gpu_available() or not enabled:
            from contextlib import nullcontext
            return nullcontext()
        
        if self.device_type == "cuda":
            return torch.cuda.amp.autocast()
        elif self.device_type == "rocm":
            # ROCm也支持CUDA AMP API
            return torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            return nullcontext()
    
    def get_grad_scaler(self, enabled: bool = True):
        """获取梯度缩放器"""
        if not self.is_gpu_available() or not enabled:
            return None
        
        if self.device_type in ["cuda", "rocm"]:
            return torch.cuda.amp.GradScaler(enabled=enabled)
        else:
            return None
    
    def empty_cache(self):
        """清空GPU缓存"""
        if self.device_type in ["cuda", "rocm"]:
            torch.cuda.empty_cache()
    
    def synchronize(self):
        """同步GPU操作"""
        if self.device_type in ["cuda", "rocm"]:
            torch.cuda.synchronize()
    
    def get_memory_info(self, device_id: int = 0) -> Tuple[int, int]:
        """获取GPU内存信息 (已用, 总计)

        Returns:
            Tuple[int, int]: (已用内存字节数, 总内存字节数)
            对于DirectML，返回(0, 0)因为不支持内存查询
        """
        if not self.is_gpu_available():
            return 0, 0

        try:
            if self.device_type in ["cuda", "rocm"]:
                allocated = torch.cuda.memory_allocated(device_id)
                reserved = torch.cuda.memory_reserved(device_id)
                return allocated, reserved
            elif self.device_type == "directml":
                # DirectML不支持内存查询，返回默认值
                return 0, 0
            else:
                # 其他设备类型
                return 0, 0
        except Exception as e:
            warnings.warn(f"获取设备 {device_id} 内存信息失败: {e}")
            return 0, 0
    
    def get_distributed_backend(self) -> str:
        """获取分布式训练后端"""
        return self.backend_type
    
    def print_device_info(self):
        """打印设备信息"""
        print(f"🔧 设备类型: {self.device_type.upper()}")
        if self.is_gpu_available():
            device_count = self.get_device_count()
            print(f"🔧 GPU数量: {device_count}")
            print(f"🔧 分布式后端: {self.backend_type}")

            # 打印每个GPU的详细信息
            for i in range(min(device_count, 4)):  # 最多显示4个GPU
                try:
                    if self.device_type == "directml":
                        # DirectML设备信息
                        print(f"🔧 GPU {i}: DirectML设备 (AMD GPU)")
                        print(f"   类型: AMD GPU via DirectML")
                    else:
                        # CUDA/ROCm设备信息
                        device_name = torch.cuda.get_device_name(i)
                        allocated, reserved = self.get_memory_info(i)
                        print(f"🔧 GPU {i}: {device_name}")
                        print(f"   内存: {allocated/1024**3:.1f}GB / {reserved/1024**3:.1f}GB")
                except Exception as e:
                    print(f"🔧 GPU {i}: 信息获取失败 ({e})")

            if self.device_type == "cuda":
                print(f"🔧 CUDA版本: {torch.version.cuda}")
                print(f"🔧 cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
            elif self.device_type == "rocm":
                hip_version = getattr(torch.version, 'hip', 'Unknown')
                print(f"🔧 ROCm/HIP版本: {hip_version}")
                # 检查ROCm特定功能
                self._print_rocm_capabilities()
        else:
            print("🔧 使用CPU模式")
            print(f"🔧 CPU线程数: {torch.get_num_threads()}")

    def _print_rocm_capabilities(self):
        """打印ROCm特定的功能支持情况"""
        try:
            # 检查Flash Attention支持
            has_flash = is_flash_attention_available()
            print(f"🔧 Flash Attention: {'支持' if has_flash else '不支持'}")

            # 检查混合精度支持
            print(f"🔧 自动混合精度: 支持")

            # 检查分布式训练支持
            print(f"🔧 分布式训练: {self.backend_type}")

        except Exception as e:
            print(f"🔧 ROCm功能检查失败: {e}")


# 全局设备管理器实例
device_manager = DeviceManager()


def get_optimal_device(prefer_device: Optional[str] = None) -> str:
    """获取最优设备配置"""
    if prefer_device and prefer_device != "auto":
        return prefer_device
    
    return device_manager.get_default_device()


def get_distributed_backend():
    """获取分布式训练后端类型

    Returns:
        str: 分布式后端类型 ('nccl', 'gloo', 'mpi')
    """
    if not device_manager.is_gpu_available():
        warnings.warn("GPU不可用，分布式训练将使用CPU后端")

    return device_manager.get_distributed_backend()


def is_flash_attention_available() -> bool:
    """检查Flash Attention是否可用"""
    # 检查PyTorch版本和Flash Attention支持
    has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    if not has_sdpa:
        return False
    
    # 对于AMD GPU，需要额外检查
    if device_manager.device_type == "rocm":
        # ROCm 5.4+ 支持Flash Attention
        try:
            rocm_version = getattr(torch.version, 'hip', None)
            if rocm_version:
                # 解析版本号并检查是否 >= 5.4
                import re
                version_match = re.search(r'(\d+)\.(\d+)', rocm_version)
                if version_match:
                    major, minor = map(int, version_match.groups())
                    if major > 5 or (major == 5 and minor >= 4):
                        return True
                    else:
                        warnings.warn(f"ROCm版本 {rocm_version} 可能不支持Flash Attention，建议升级到5.4+")
                        return False
                # 如果无法解析版本，保守返回True
                return True
        except (AttributeError, RuntimeError):
            pass
    
    return has_sdpa


def get_device_specific_config() -> dict:
    """获取设备特定的配置"""
    config = {
        "device_type": device_manager.device_type,
        "supports_amp": device_manager.is_gpu_available(),
        "supports_flash_attention": is_flash_attention_available(),
        "distributed_backend": device_manager.get_distributed_backend(),
        "device_count": device_manager.get_device_count()
    }
    
    return config
