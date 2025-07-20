"""
Flash Attention 兼容性测试脚本
测试在不同GPU平台上的Flash Attention性能和兼容性
"""

import torch
import torch.nn.functional as F
import time
import warnings
from utils.device_utils import device_manager, get_device_specific_config


def test_cpu_sdpa_compatibility():
    """在CPU环境下测试基础SDPA功能"""

    print("🔍 CPU环境SDPA兼容性测试")
    print("-" * 30)

    try:
        import torch.nn.functional as F

        # 检查SDPA是否可用
        has_sdpa = hasattr(F, 'scaled_dot_product_attention')
        print(f"SDPA可用: {'✅' if has_sdpa else '❌'}")

        if not has_sdpa:
            print("❌ PyTorch版本不支持scaled_dot_product_attention")
            return False

        # 创建CPU测试数据
        batch_size, num_heads, seq_len, head_dim = 1, 4, 32, 16  # 较小的尺寸用于CPU测试

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)

        print(f"测试张量形状: {query.shape}")

        # 测试基础SDPA功能
        with torch.no_grad():
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )

        print(f"✅ SDPA基础功能测试成功")
        print(f"输出形状: {output.shape}")
        print(f"输出数值范围: [{output.min():.3f}, {output.max():.3f}]")

        # 验证输出的合理性
        if torch.isnan(output).any():
            print("❌ 输出包含NaN值")
            return False

        if torch.isinf(output).any():
            print("❌ 输出包含无穷值")
            return False

        print("✅ CPU环境SDPA兼容性测试通过")
        return True

    except Exception as e:
        print(f"❌ CPU SDPA测试失败: {e}")
        return False


def test_flash_attention_compatibility():
    """测试Flash Attention在当前设备上的兼容性"""

    print("🔍 Flash Attention 兼容性测试")
    print("=" * 50)

    # 获取设备配置
    device_config = get_device_specific_config()
    device = device_manager.get_device_object()

    print(f"设备类型: {device_config['device_type']}")
    print(f"设备: {device}")
    print(f"支持Flash Attention: {device_config['supports_flash_attention']}")

    # 详细的环境信息
    print(f"PyTorch版本: {torch.__version__}")
    if device_config['device_type'] == 'rocm':
        hip_version = getattr(torch.version, 'hip', 'Unknown')
        print(f"HIP版本: {hip_version}")
    elif device_config['device_type'] == 'cuda':
        print(f"CUDA版本: {torch.version.cuda}")

    if not device_manager.is_gpu_available():
        print("⚠️ CPU环境，将测试基础SDPA功能")
        # 在CPU环境下测试基础SDPA功能
        return test_cpu_sdpa_compatibility()
    
    # 测试参数
    batch_size = 2
    seq_len = 512
    num_heads = 8
    head_dim = 64
    
    try:
        # 创建测试张量
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        print(f"\n📊 测试张量形状: {query.shape}")
        
        # 测试1: 检查SDPA是否可用
        has_sdpa = hasattr(F, 'scaled_dot_product_attention')
        print(f"PyTorch SDPA 可用: {has_sdpa}")
        
        if not has_sdpa:
            print("❌ PyTorch版本不支持scaled_dot_product_attention")
            return False
        
        # 测试2: 基础SDPA功能测试
        print("\n🧪 测试基础SDPA功能...")
        start_time = time.time()
        
        with torch.no_grad():
            output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )
        
        device_manager.synchronize()
        sdpa_time = time.time() - start_time
        
        print(f"✅ SDPA测试成功")
        print(f"输出形状: {output.shape}")
        print(f"执行时间: {sdpa_time:.4f}s")
        
        # 测试3: 详细的后端兼容性测试
        print("\n🚀 测试各种注意力后端...")

        backend_results = {}

        # 测试Flash Attention后端
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False
            ):
                start_time = time.time()
                with torch.no_grad():
                    flash_output = F.scaled_dot_product_attention(
                        query, key, value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True
                    )
                device_manager.synchronize()
                flash_time = time.time() - start_time

                # 验证输出一致性
                diff = torch.abs(output - flash_output).max().item()

                backend_results['flash'] = {
                    'success': True,
                    'time': flash_time,
                    'speedup': sdpa_time/flash_time,
                    'max_diff': diff,
                    'accurate': diff < 1e-3
                }

                print(f"✅ Flash Attention: {flash_time:.4f}s (提升{sdpa_time/flash_time:.2f}x)")

        except Exception as e:
            backend_results['flash'] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Flash Attention失败: {e}")

        # 测试Memory Efficient后端
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True
            ):
                start_time = time.time()
                with torch.no_grad():
                    mem_eff_output = F.scaled_dot_product_attention(
                        query, key, value,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True
                    )
                device_manager.synchronize()
                mem_eff_time = time.time() - start_time

                diff = torch.abs(output - mem_eff_output).max().item()

                backend_results['memory_efficient'] = {
                    'success': True,
                    'time': mem_eff_time,
                    'speedup': sdpa_time/mem_eff_time,
                    'max_diff': diff,
                    'accurate': diff < 1e-3
                }

                print(f"✅ Memory Efficient: {mem_eff_time:.4f}s (提升{sdpa_time/mem_eff_time:.2f}x)")

        except Exception as e:
            backend_results['memory_efficient'] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Memory Efficient失败: {e}")

        # 分析结果
        successful_backends = [name for name, result in backend_results.items()
                             if result.get('success', False)]

        if successful_backends:
            print(f"\n✅ 可用后端: {', '.join(successful_backends)}")

            # 推荐最佳后端
            best_backend = None
            best_time = float('inf')

            for name, result in backend_results.items():
                if result.get('success') and result.get('accurate', False):
                    if result['time'] < best_time:
                        best_time = result['time']
                        best_backend = name

            if best_backend:
                print(f"🏆 推荐后端: {best_backend}")
                return True
            else:
                print("⚠️ 没有找到准确且高效的后端")
                return False
        else:
            print("❌ 所有优化后端都不可用，回退到数学实现")
            
            # 测试4: 回退到数学实现
            print("\n🔢 测试数学后端...")
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False
                ):
                    start_time = time.time()
                    with torch.no_grad():
                        math_output = F.scaled_dot_product_attention(
                            query, key, value,
                            attn_mask=None,
                            dropout_p=0.0,
                            is_causal=True
                        )
                    device_manager.synchronize()
                    math_time = time.time() - start_time
                    
                    print(f"✅ 数学后端测试成功")
                    print(f"执行时间: {math_time:.4f}s")
                    return True
                    
            except Exception as e2:
                print(f"❌ 数学后端测试也失败: {e2}")
                return False
    
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        return False


def benchmark_attention_backends():
    """对比不同注意力后端的性能"""
    
    if not device_manager.is_gpu_available():
        print("❌ 需要GPU进行性能测试")
        return
    
    print("\n🏁 注意力机制性能对比")
    print("=" * 50)
    
    device = device_manager.get_device_object()
    
    # 测试参数
    configs = [
        (1, 512, 8, 64),   # 小规模
        (2, 1024, 8, 64),  # 中等规模
        (4, 2048, 8, 64),  # 大规模
    ]
    
    for batch_size, seq_len, num_heads, head_dim in configs:
        print(f"\n📏 配置: B={batch_size}, L={seq_len}, H={num_heads}, D={head_dim}")
        
        # 创建测试数据
        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        backends = [
            ("Flash", True, False, False),
            ("Memory Efficient", False, False, True),
            ("Math", False, True, False),
        ]
        
        for name, flash, math, mem_eff in backends:
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=flash,
                    enable_math=math,
                    enable_mem_efficient=mem_eff
                ):
                    # 预热
                    for _ in range(3):
                        with torch.no_grad():
                            _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
                    
                    device_manager.synchronize()
                    
                    # 性能测试
                    start_time = time.time()
                    for _ in range(10):
                        with torch.no_grad():
                            _ = F.scaled_dot_product_attention(query, key, value, is_causal=True)
                    
                    device_manager.synchronize()
                    avg_time = (time.time() - start_time) / 10
                    
                    print(f"  {name:15}: {avg_time*1000:.2f}ms")
                    
            except Exception as e:
                print(f"  {name:15}: 不支持 ({str(e)[:30]}...)")


def get_attention_recommendations():
    """获取当前设备的注意力机制推荐配置"""
    
    device_config = get_device_specific_config()
    
    recommendations = {
        "use_flash_attention": False,
        "fallback_to_math": True,
        "dtype_recommendation": "float16",
        "notes": []
    }
    
    if device_config["device_type"] == "cuda":
        recommendations["use_flash_attention"] = True
        recommendations["notes"].append("NVIDIA GPU支持完整的Flash Attention")
        
    elif device_config["device_type"] == "rocm":
        # ROCm的Flash Attention支持取决于版本
        try:
            rocm_version = getattr(torch.version, 'hip', None)
            if rocm_version:
                recommendations["use_flash_attention"] = True
                recommendations["notes"].append(f"AMD GPU (ROCm {rocm_version}) 支持Flash Attention")
            else:
                recommendations["notes"].append("AMD GPU Flash Attention支持需要验证")
        except (AttributeError, RuntimeError):
            recommendations["notes"].append("无法确定ROCm版本，建议测试Flash Attention兼容性")
            
    else:
        recommendations["use_flash_attention"] = False
        recommendations["fallback_to_math"] = True
        recommendations["dtype_recommendation"] = "float32"
        recommendations["notes"].append("CPU模式，使用标准数学实现")
    
    return recommendations


if __name__ == "__main__":
    print("🧪 MiniMind Flash Attention 兼容性测试")
    print("=" * 60)
    
    # 基础兼容性测试
    is_compatible = test_flash_attention_compatibility()
    
    # 性能对比测试
    if is_compatible:
        benchmark_attention_backends()
    
    # 获取推荐配置
    print("\n💡 推荐配置")
    print("=" * 50)
    recommendations = get_attention_recommendations()
    
    for key, value in recommendations.items():
        if key != "notes":
            print(f"{key}: {value}")
    
    print("\n📝 注意事项:")
    for note in recommendations["notes"]:
        print(f"  • {note}")
