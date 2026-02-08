#!/usr/bin/env python3
"""
Command Line Interface
命令行接口
提供系统的主要命令行操作入口
"""

import argparse
import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from main_pipeline import ArchiveAnalysisPipeline
from config import load_config
from usage_examples import main as run_examples


def analyze_archive(args):
    """分析档案命令"""
    print(f"开始分析档案: {args.archive_path}")
    
    # 初始化流水线
    pipeline = ArchiveAnalysisPipeline(
        api_key=args.api_key,
        output_dir=args.output_dir
    )
    
    try:
        results = pipeline.process_archive(args.archive_path)
        
        print("\n=== 分析完成 ===")
        print(f"处理文档数: {results['pipeline_statistics']['processed_documents']}")
        print(f"处理图片数: {results['pipeline_statistics']['processed_images']}")
        print(f"识别实体数: {results['pipeline_statistics']['total_entities']}")
        print(f"抽取关系数: {results['pipeline_statistics']['total_relations']}")
        print(f"Token消耗: {results['pipeline_statistics']['total_tokens_consumed']}")
        
        if args.verbose:
            print(f"\n详细结果保存在: {args.output_dir}")
            
    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        sys.exit(1)


def run_demo(args):
    """运行演示命令"""
    print("运行系统演示...")
    run_examples()


def validate_setup(args):
    """验证环境配置命令"""
    print("验证系统配置...")
    
    # 检查API密钥
    api_key = args.api_key or os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ 未找到API密钥")
        print("请设置 DEEPSEEK_API_KEY 环境变量或使用 --api-key 参数")
        return False
    
    print("✅ API密钥已设置")
    
    # 检查依赖包
    required_packages = ['PIL', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install pillow numpy")
        return False
    
    print("✅ 环境配置验证通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Epstein档案分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s analyze ./archive --api-key YOUR_KEY
  %(prog)s demo
  %(prog)s validate --api-key YOUR_KEY
        """
    )
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析档案')
    analyze_parser.add_argument('archive_path', help='档案目录路径')
    analyze_parser.add_argument('--api-key', help='DeepSeek API密钥')
    analyze_parser.add_argument('--output-dir', default='./results', 
                               help='输出目录 (默认: ./results)')
    analyze_parser.add_argument('--verbose', '-v', action='store_true',
                               help='详细输出')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证环境配置')
    validate_parser.add_argument('--api-key', help='DeepSeek API密钥')
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行相应命令
    if args.command == 'analyze':
        if not validate_setup(args):
            sys.exit(1)
        analyze_archive(args)
    elif args.command == 'demo':
        run_demo(args)
    elif args.command == 'validate':
        validate_setup(args)


if __name__ == "__main__":
    main()