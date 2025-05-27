"""
 * Description: Simple demo for benchmarking GEMM (matrix multiplication) and GEMV (matrix-vector multiplication) roofline.
 * Usage: ncu --set full -o ncu_profile_simple python3 ncu_profile_simple.py --m 4096 --n 4096 --k 4096
 * Date: 2025-05-26
"""
import torch
import time
import argparse
from typing import Tuple

def benchmark_gemm(
    m: int, 
    n: int, 
    k: int, 
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
    num_warmups: int = 10,
    num_iters: int = 100
) -> Tuple[float, float]:
    """
    基准测试 GEMM (矩阵乘法) 操作
    
    参数:
        m, n, k: 矩阵维度 (A: m×k, B: k×n, C: m×n)
        dtype: 数据类型
        device: 计算设备 ('cuda' 或 'cpu')
        num_warmups: 预热迭代次数
        num_iters: 测量迭代次数
    
    返回:
        (平均时间(ms), 计算吞吐量(TFLOPs))
    """
    # 创建随机矩阵
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)
    
    # 预热
    for _ in range(num_warmups):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    # 基准测试
    start_time = time.time()
    for _ in range(num_iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    
    # 计算吞吐量 (TFLOPs)
    flops = 2 * m * n * k  # 乘加算作2次浮点运算
    tflops = flops / (elapsed_ms * 1e9)  # 转换为 TFLOPs
    
    return elapsed_ms, tflops

def benchmark_gemv(
    m: int, 
    n: int, 
    dtype: torch.dtype = torch.float32,
    device: str = 'cuda',
    num_warmups: int = 10,
    num_iters: int = 10
) -> Tuple[float, float]:
    """
    基准测试 GEMV (矩阵-向量乘法) 操作
    
    参数:
        m, n: 矩阵维度 (A: m×n, x: n×1, y: m×1)
        dtype: 数据类型
        device: 计算设备 ('cuda' 或 'cpu')
        num_warmups: 预热迭代次数
        num_iters: 测量迭代次数
    
    返回:
        (平均时间(ms), 计算吞吐量(TFLOPs))
    """
    # 创建随机矩阵和向量
    a = torch.randn(m, n, dtype=dtype, device=device)
    x = torch.randn(n, dtype=dtype, device=device)
    
    # 预热
    for _ in range(num_warmups):
        y = torch.matmul(a, x)
        torch.cuda.synchronize()
    
    # 基准测试
    start_time = time.time()
    for _ in range(num_iters):
        y = torch.matmul(a, x)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000 / num_iters
    
    # 计算吞吐量 (TFLOPs)
    flops = 2 * m * n  # 乘加算作2次浮点运算
    tflops = flops / (elapsed_ms * 1e9)  # 转换为 TFLOPs
    
    return elapsed_ms, tflops

def main():
    parser = argparse.ArgumentParser(description='PyTorch GEMM/GEMV 基准测试工具')
    # parser.add_argument('--op', type=str, choices=['gemm', 'gemv'], required=True,
    #                    help='要测试的操作: gemm (矩阵乘法) 或 gemv (矩阵-向量乘法)')
    parser.add_argument('--m', type=int, required=True, help='矩阵的行数')
    parser.add_argument('--n', type=int, required=True, help='矩阵的列数')
    parser.add_argument('--k', type=int, default=None, 
                       help='对于GEMM, 内部维度 (A: m×k, B: k×n)')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='数据类型')
    parser.add_argument('--warmups', type=int, default=10,
                       help='预热迭代次数')
    parser.add_argument('--iters', type=int, default=10,
                       help='测量迭代次数')
    
    args = parser.parse_args()
    
    # 设置数据类型
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA不可用，请在支持CUDA的环境中运行')
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"设备: {torch.cuda.get_device_name(0)}")
    print(f"操作数据类型: {args.dtype}")
    
    # gemm
    if args.k is None:
        raise ValueError('对于GEMM操作，必须提供--k参数')
    print(f"矩阵维度: A[{args.m}×{args.k}] * B[{args.k}×{args.n}]")
    elapsed_ms, tflops = benchmark_gemm(
        args.m, args.n, args.k, dtype, 'cuda', args.warmups, args.iters
    )
    print(f"平均耗时: {elapsed_ms:.3f} ms")
    print(f"计算吞吐量: {tflops:.3f} TFLOPs")

    # gemv
    print(f"矩阵维度: A[{args.m * 4}×{args.n * 4}] * x[{args.n * 4}×1]")
    elapsed_ms, tflops = benchmark_gemv(
        args.m * 4, args.n * 4, dtype, 'cuda', args.warmups, args.iters
    )
    
    print(f"平均耗时: {elapsed_ms:.3f} ms")
    print(f"计算吞吐量: {tflops:.3f} TFLOPs")

if __name__ == '__main__':
    main()

