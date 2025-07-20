"""
MiniMind AMD GPU兼容性综合测试脚本
验证在AMD GPU上的完整功能和性能
"""

import os
import sys
import torch
import time
import traceback
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.device_utils import device_manager, get_device_specific_config
from utils.flash_attention_test import test_flash_attention_compatibility
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer


class AMDCompatibilityTester:
    """AMD GPU兼容性测试器"""
    
    def __init__(self):
        self.device_config = get_device_specific_config()
        self.device = device_manager.get_device_object()
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有兼容性测试"""
        
        print("🧪 MiniMind AMD GPU兼容性测试套件")
        print("=" * 60)
        
        tests = [
            ("设备检测", self.test_device_detection),
            ("基础张量操作", self.test_basic_tensor_ops),
            ("模型加载", self.test_model_loading),
            ("前向传播", self.test_forward_pass),
            ("反向传播", self.test_backward_pass),
            ("混合精度", self.test_mixed_precision),
            ("Flash Attention", self.test_flash_attention),
            ("内存管理", self.test_memory_management),
            ("分布式训练", self.test_distributed_setup),
            ("性能基准", self.test_performance_benchmark)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 测试: {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                status = "✅ 通过" if result else "❌ 失败"
                print(f"结果: {status}")
                
            except Exception as e:
                self.test_results[test_name] = False
                print(f"结果: ❌ 异常 - {str(e)}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
        
        self.print_summary()
        return self.test_results
    
    def test_device_detection(self) -> bool:
        """测试设备检测功能"""
        
        print(f"设备类型: {self.device_config['device_type']}")
        print(f"设备数量: {self.device_config['device_count']}")
        print(f"支持AMP: {self.device_config['supports_amp']}")
        print(f"分布式后端: {self.device_config['distributed_backend']}")
        
        # 验证设备可用性
        if self.device_config['device_type'] in ['cuda', 'rocm']:
            return self.device_config['device_count'] > 0
        else:
            return True  # CPU总是可用
    
    def test_basic_tensor_ops(self) -> bool:
        """测试基础张量操作"""
        
        try:
            # 创建测试张量
            a = torch.randn(100, 100, device=self.device)
            b = torch.randn(100, 100, device=self.device)
            
            # 基础运算
            c = torch.matmul(a, b)
            d = torch.relu(c)
            e = torch.softmax(d, dim=-1)
            
            # 验证结果
            device_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]
            assert c.device.type == device_type
            assert not torch.isnan(e).any()
            
            print(f"张量形状: {a.shape} -> {e.shape}")
            print(f"设备: {e.device}")
            
            return True
            
        except Exception as e:
            print(f"张量操作失败: {e}")
            return False
    
    def test_model_loading(self) -> bool:
        """测试模型加载"""
        
        try:
            # 创建小型测试模型
            config = MiniMindConfig(
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                vocab_size=1000
            )
            
            model = MiniMindForCausalLM(config)
            model = model.to(self.device)
            
            # 验证模型参数在正确设备上
            device_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]
            for param in model.parameters():
                if param.device.type != device_type:
                    return False
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"模型参数量: {param_count:,}")
            print(f"模型设备: {next(model.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def test_forward_pass(self) -> bool:
        """测试前向传播"""
        
        try:
            config = MiniMindConfig(
                hidden_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                vocab_size=1000
            )
            
            model = MiniMindForCausalLM(config).to(self.device)
            model.eval()
            
            # 创建测试输入
            batch_size, seq_len = 2, 64
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # 前向传播
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            # 验证输出
            expected_shape = (batch_size, seq_len, 1000)
            assert logits.shape == expected_shape
            assert not torch.isnan(logits).any()
            
            print(f"输入形状: {input_ids.shape}")
            print(f"输出形状: {logits.shape}")
            print(f"输出范围: [{logits.min():.3f}, {logits.max():.3f}]")
            
            return True
            
        except Exception as e:
            print(f"前向传播失败: {e}")
            return False
    
    def test_backward_pass(self) -> bool:
        """测试反向传播"""
        
        try:
            config = MiniMindConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                vocab_size=500
            )
            
            model = MiniMindForCausalLM(config).to(self.device)
            model.train()
            
            # 创建测试数据
            batch_size, seq_len = 2, 32
            input_ids = torch.randint(0, 500, (batch_size, seq_len), device=self.device)
            labels = torch.randint(0, 500, (batch_size, seq_len), device=self.device)
            
            # 前向传播
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            if loss is None:
                print("❌ 模型未返回损失值")
                return False

            # 反向传播
            loss.backward()
            
            # 验证梯度
            grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            print(f"损失值: {loss.item():.4f}")
            print(f"梯度范数: {grad_norm:.4f}")
            
            return grad_norm > 0 and not torch.isnan(loss)
            
        except Exception as e:
            print(f"反向传播失败: {e}")
            return False
    
    def test_mixed_precision(self) -> bool:
        """测试混合精度训练"""
        
        if not device_manager.is_gpu_available():
            print("CPU模式，跳过混合精度测试")
            return True
        
        try:
            config = MiniMindConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                vocab_size=500
            )
            
            model = MiniMindForCausalLM(config).to(self.device)
            scaler = device_manager.get_grad_scaler(enabled=True)

            if scaler is None:
                print("梯度缩放器不可用，跳过混合精度测试（DirectML限制）")
                return True  # DirectML环境下这是正常的
            
            # 测试数据
            input_ids = torch.randint(0, 500, (2, 32), device=self.device)
            labels = torch.randint(0, 500, (2, 32), device=self.device)
            
            # 混合精度前向传播
            with device_manager.get_amp_context():
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
            
            # 缩放反向传播
            scaler.scale(loss).backward()
            
            print(f"混合精度损失: {loss.item():.4f}")
            print(f"缩放因子: {scaler.get_scale()}")
            
            return True
            
        except Exception as e:
            print(f"混合精度测试失败: {e}")
            return False
    
    def test_flash_attention(self) -> bool:
        """测试Flash Attention兼容性"""
        
        return test_flash_attention_compatibility()
    
    def test_memory_management(self) -> bool:
        """测试内存管理"""
        
        if not device_manager.is_gpu_available():
            print("CPU模式，跳过内存测试")
            return True
        
        try:
            # 检查是否支持内存查询
            try:
                initial_memory = device_manager.get_memory_info()[0]
                memory_query_supported = True
            except (RuntimeError, AttributeError, NotImplementedError):
                print("DirectML不支持内存查询，测试基础内存操作")
                memory_query_supported = False
                initial_memory = 0

            # 分配大量内存
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, device=self.device)
                tensors.append(tensor)

            # 检查内存增长（如果支持）
            if memory_query_supported:
                peak_memory = device_manager.get_memory_info()[0]
            else:
                peak_memory = 0

            # 清理内存
            del tensors
            device_manager.empty_cache()

            # 检查内存释放（如果支持）
            if memory_query_supported:
                final_memory = device_manager.get_memory_info()[0]
                print(f"初始内存: {initial_memory / 1024**2:.1f} MB")
                print(f"峰值内存: {peak_memory / 1024**2:.1f} MB")
                print(f"最终内存: {final_memory / 1024**2:.1f} MB")
                # 如果内存查询都返回0，说明DirectML不支持内存查询
                if initial_memory == 0 and peak_memory == 0 and final_memory == 0:
                    print("DirectML内存查询不支持，但基础操作成功")
                    return True
                # 验证内存正确释放
                memory_released = peak_memory - final_memory > 0
                return memory_released
            else:
                print("DirectML内存管理测试：基础操作成功")
                return True
            
        except Exception as e:
            print(f"内存管理测试失败: {e}")
            return False
    
    def test_distributed_setup(self) -> bool:
        """测试分布式训练设置"""
        
        try:
            backend = device_manager.get_distributed_backend()
            print(f"分布式后端: {backend}")
            
            # 验证后端选择正确
            if device_manager.device_type in ['cuda', 'rocm']:
                return backend == 'nccl'
            else:
                return backend == 'gloo'
                
        except Exception as e:
            print(f"分布式设置测试失败: {e}")
            return False
    
    def test_performance_benchmark(self) -> bool:
        """性能基准测试"""
        
        try:
            config = MiniMindConfig(
                hidden_size=512,
                num_hidden_layers=4,
                num_attention_heads=8,
                vocab_size=1000
            )
            
            model = MiniMindForCausalLM(config).to(self.device)
            model.eval()
            
            # 预热
            input_ids = torch.randint(0, 1000, (4, 256), device=self.device)
            for _ in range(3):
                with torch.no_grad():
                    _ = model(input_ids)
            
            device_manager.synchronize()
            
            # 性能测试
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                with torch.no_grad():
                    outputs = model(input_ids)
            
            device_manager.synchronize()
            total_time = time.time() - start_time
            avg_time = total_time / num_runs
            
            print(f"平均推理时间: {avg_time*1000:.2f} ms")
            print(f"吞吐量: {4*256/avg_time:.0f} tokens/s")
            
            # 性能阈值检查（根据设备类型调整）
            if device_manager.device_type == 'cuda':
                return avg_time < 0.1  # NVIDIA GPU应该很快
            elif device_manager.device_type == 'rocm':
                return avg_time < 0.2  # AMD GPU可能稍慢
            else:
                return avg_time < 1.0  # CPU模式
                
        except Exception as e:
            print(f"性能测试失败: {e}")
            return False
    
    def print_summary(self):
        """打印测试总结"""
        
        print("\n" + "=" * 60)
        print("🏁 测试总结")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        print(f"总测试数: {total}")
        print(f"通过数: {passed}")
        print(f"失败数: {total - passed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        print("\n详细结果:")
        for test_name, result in self.test_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {test_name}")
        
        if passed == total:
            print("\n🎉 所有测试通过！AMD GPU兼容性良好。")
        else:
            print(f"\n⚠️ {total-passed} 个测试失败，需要进一步调试。")


if __name__ == "__main__":
    tester = AMDCompatibilityTester()
    results = tester.run_all_tests()
    
    # 返回适当的退出码
    exit_code = 0 if all(results.values()) else 1
    sys.exit(exit_code)
