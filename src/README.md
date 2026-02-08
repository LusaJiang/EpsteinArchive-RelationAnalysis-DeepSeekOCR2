# Epstein Archive Analysis System

基于DeepSeek OCR2的爱泼斯坦档案人物关系分析系统

## 系统概述

本系统实现了论文《基于LLM视觉模型的爱泼斯坦档案人物关系分析可行性研究》中提出的完整分析流程，包含以下核心功能：

1. **OCR文本化处理** - 使用DeepSeek OCR2将扫描件转换为结构化文本
2. **文本关系抽取** - 利用DeepSeek V3.2大语言模型进行命名实体识别和关系抽取
3. **视觉要素分析** - 借助DeepSeek OCR2的视觉功能进行图片描述和要素识别
4. **多模态关联分析** - 整合文本和图片信息进行综合人物关系分析

## 目录结构

```
src/
├── __init__.py              # 包初始化文件
├── config.py               # 系统配置模块
├── ocr_processor.py        # OCR文本处理器
├── llm_analyzer.py         # LLM文本分析器
├── vision_analyzer.py      # 视觉分析器
├── main_pipeline.py        # 主分析流水线
└── usage_examples.py       # 使用示例
```

## 核心模块介绍

### 1. OCR处理器 (`ocr_processor.py`)

实现基于DeepSeek OCR2的文档扫描件文本化处理功能。

**主要特性：**
- 支持多种图像格式（PDF, JPG, PNG, TIFF等）
- 自动图像预处理（旋转校正、亮度调整）
- 批量处理能力
- 处理统计和质量评估

**使用示例：**
```python
from ocr_processor import DeepSeekOCRProcessor

# 初始化处理器
ocr = DeepSeekOCRProcessor(api_key="your_api_key")

# 单文件处理
result = ocr.recognize_text("./document.jpg")

# 批量处理
stats = ocr.batch_process("./input_folder", "./output_folder")
```

### 2. LLM分析器 (`llm_analyzer.py`)

基于DeepSeek V3.2实现文本分析和人物关系抽取。

**主要特性：**
- 命名实体识别（人物、地点、组织、事件）
- 多类型关系抽取（社交、商业、政治、家庭等）
- 基于规则和深度学习的混合方法
- 批量文档分析

**使用示例：**
```python
from llm_analyzer import DeepSeekLLMAnalyzer

# 初始化分析器
analyzer = DeepSeekLLMAnalyzer(api_key="your_api_key")

# 分析文本
text = "Jeffrey Epstein invited Donald Trump to Lolita Island..."
result = analyzer.analyze_document(text)

# 批量分析
results = analyzer.batch_analyze(["./doc1.txt", "./doc2.txt"])
```

### 3. 视觉分析器 (`vision_analyzer.py`)

利用DeepSeek OCR2的视觉功能进行图片分析。

**主要特性：**
- 图片内容自动描述
- 要素识别与提取（人物、场景、物品、动作）
- 人物身份识别与匹配
- 结合文本上下文的关联分析

**使用示例：**
```python
from vision_analyzer import DeepSeekVisionAnalyzer

# 初始化分析器
vision = DeepSeekVisionAnalyzer(api_key="your_api_key")

# 图片描述
description = vision.describe_image("./photo.jpg")

# 要素识别
elements = vision.identify_elements("./photo.jpg", context_text="相关文本")

# 综合分析
result = vision.analyze_with_context("./photo.jpg", "2018年3月的会议记录")
```

### 4. 主流水线 (`main_pipeline.py`)

整合所有模块的完整分析流程。

**四阶段分析流程：**
1. OCR文本化处理
2. 文本关系抽取  
3. 图片要素分析
4. 多模态关联整合

**使用示例：**
```python
from main_pipeline import ArchiveAnalysisPipeline

# 初始化流水线
pipeline = ArchiveAnalysisPipeline(
    api_key="your_api_key",
    output_dir="./results"
)

# 处理完整档案
results = pipeline.process_archive("./epstein_archive")
```

## 配置说明 (`config.py`)

系统配置模块定义了各个组件的参数设置：

- **OCR配置**：模型选择、语言设置、置信度阈值
- **LLM配置**：最大token数、温度参数、关系类型
- **视觉配置**：描述详细程度、检测阈值
- **处理配置**：批量大小、并发请求数、超时设置
- **分析配置**：核心人物列表、关键地点、敏感词过滤

## 使用示例 (`usage_examples.py`)

提供了完整的使用示例，包括：

- 各模块独立使用示例
- 完整流水线运行示例
- 自定义分析流程示例
- 配置管理和最佳实践

## 安装要求

```bash
pip install pillow numpy
```

## API密钥设置

系统支持多种方式设置API密钥：

1. 构造函数参数传递
2. 环境变量 `DEEPSEEK_API_KEY`
3. 配置文件设置

## 注意事项

1. **数据隐私**：处理敏感档案时请注意数据安全和隐私保护
2. **API限制**：注意DeepSeek API的调用频率和token消耗限制
3. **文件格式**：确保输入文件格式符合支持的类型
4. **错误处理**：建议在生产环境中添加完善的异常处理机制

## 性能指标

根据论文研究结果，系统预期性能：
- OCR识别准确率：97.1%
- 文本关系抽取覆盖率：显著优于传统方法
- 图片要素识别准确率：91.5%
- 整体处理效率：大幅提升

## 扩展开发

系统采用模块化设计，便于扩展：

- 可轻松集成新的OCR引擎
- 支持添加自定义关系类型
- 可扩展视觉分析功能
- 支持插件化分析模块

## 免责声明

本系统仅用于学术研究和技术验证，请遵守相关法律法规，谨慎处理敏感数据。