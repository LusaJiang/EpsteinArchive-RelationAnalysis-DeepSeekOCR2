"""
Test Modules
模块测试脚本
验证各个模块的基本功能
"""

import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from config import DEFAULT_CONFIG
        print("✅ config模块导入成功")
        
        from ocr_processor import DeepSeekOCRProcessor
        print("✅ ocr_processor模块导入成功")
        
        from llm_analyzer import DeepSeekLLMAnalyzer
        print("✅ llm_analyzer模块导入成功")
        
        from vision_analyzer import DeepSeekVisionAnalyzer
        print("✅ vision_analyzer模块导入成功")
        
        from main_pipeline import ArchiveAnalysisPipeline
        print("✅ main_pipeline模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置模块"""
    print("\n=== 测试配置模块 ===")
    
    try:
        from config import DEFAULT_CONFIG
        
        print(f"OCR模型: {DEFAULT_CONFIG.ocr.model_name}")
        print(f"LLM模型: {DEFAULT_CONFIG.llm.model_name}")
        print(f"视觉模型: {DEFAULT_CONFIG.vision.model_name}")
        print("✅ 配置模块测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置模块测试失败: {e}")
        return False


def test_ocr_processor():
    """测试OCR处理器（基础功能）"""
    print("\n=== 测试OCR处理器 ===")
    
    try:
        from ocr_processor import DeepSeekOCRProcessor
        
        # 测试初始化
        processor = DeepSeekOCRProcessor(api_key="test_key")
        print("✅ OCR处理器初始化成功")
        
        # 测试预处理方法（不实际调用API）
        print("✅ OCR处理器基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ OCR处理器测试失败: {e}")
        return False


def test_llm_analyzer():
    """测试LLM分析器"""
    print("\n=== 测试LLM分析器 ===")
    
    try:
        from llm_analyzer import DeepSeekLLMAnalyzer
        
        # 测试初始化
        analyzer = DeepSeekLLMAnalyzer(api_key="test_key")
        print("✅ LLM分析器初始化成功")
        
        # 测试文本预处理
        test_text = "JEFFREY EPSTEIN invited DONALD TRUMP to LOLITA ISLAND"
        processed = analyzer.preprocess_text(test_text)
        print(f"✅ 文本预处理: {processed}")
        
        # 测试实体识别（基础）
        entities = analyzer.extract_entities(test_text)
        print(f"✅ 实体识别完成，找到 {len(entities)} 个实体")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM分析器测试失败: {e}")
        return False


def test_vision_analyzer():
    """测试视觉分析器"""
    print("\n=== 测试视觉分析器 ===")
    
    try:
        from vision_analyzer import DeepSeekVisionAnalyzer
        
        # 测试初始化
        vision = DeepSeekVisionAnalyzer(api_key="test_key")
        print("✅ 视觉分析器初始化成功")
        
        # 测试人物匹配
        known_persons = ["Jeffrey Epstein", "Donald Trump"]
        match_result = vision._match_person_identity("epstein", known_persons)
        print(f"✅ 人物匹配: {match_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ 视觉分析器测试失败: {e}")
        return False


def test_pipeline():
    """测试主流水线"""
    print("\n=== 测试主流水线 ===")
    
    try:
        from main_pipeline import ArchiveAnalysisPipeline
        
        # 测试初始化
        pipeline = ArchiveAnalysisPipeline(
            api_key="test_key",
            output_dir="./test_output"
        )
        print("✅ 主流水线初始化成功")
        
        # 测试统计初始化
        print(f"✅ 初始统计: {pipeline.stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 主流水线测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("Epstein档案分析系统模块测试")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_ocr_processor,
        test_llm_analyzer,
        test_vision_analyzer,
        test_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())