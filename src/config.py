"""
Configuration Module
配置模块
定义分析系统的各种配置参数
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OCRConfig:
    """OCR配置"""
    model_name: str = "deepseek-ocr-v2"
    language: str = "multi"  # en, zh, multi
    confidence_threshold: float = 0.8
    max_resolution: int = 1024
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']


@dataclass
class LLMConfig:
    """大语言模型配置"""
    model_name: str = "deepseek-chat"
    max_tokens: int = 2000
    temperature: float = 0.3
    top_p: float = 0.9
    confidence_threshold: float = 0.7
    relation_types: List[str] = None
    
    def __post_init__(self):
        if self.relation_types is None:
            self.relation_types = [
                "social_interaction",
                "business_association", 
                "political_connection",
                "family_relationship",
                "travel_companion",
                "financial_tie"
            ]


@dataclass
class VisionConfig:
    """视觉分析配置"""
    model_name: str = "deepseek-vision"
    description_detail_level: str = "detailed"  # basic, detailed, comprehensive
    element_detection_threshold: float = 0.6
    person_similarity_threshold: float = 0.7
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']


@dataclass
class ProcessingConfig:
    """处理配置"""
    batch_size: int = 10
    max_concurrent_requests: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300
    output_format: str = "json"  # json, csv, txt


@dataclass
class AnalysisConfig:
    """分析配置"""
    core_persons: List[str] = None
    key_locations: List[str] = None
    sensitive_keywords: List[str] = None
    analysis_depth: str = "comprehensive"  # basic, detailed, comprehensive
    
    def __post_init__(self):
        if self.core_persons is None:
            self.core_persons = [
                "Jeffrey Epstein",
                "Donald Trump", 
                "Bill Clinton",
                "Elon Musk",
                "Prince Andrew",
                "Ghislaine Maxwell",
                "Barack Obama"
            ]
        
        if self.key_locations is None:
            self.key_locations = [
                "Lolita Island",
                "Palm Beach",
                "New York",
                "Washington D.C."
            ]
        
        if self.sensitive_keywords is None:
            self.sensitive_keywords = [
                "minor", "underage", "illegal", "bribery", "corruption"
            ]


@dataclass
class SystemConfig:
    """系统配置"""
    ocr: OCRConfig
    llm: LLMConfig  
    vision: VisionConfig
    processing: ProcessingConfig
    analysis: AnalysisConfig
    log_level: str = "INFO"
    output_directory: str = "./analysis_results"


# 默认配置实例
DEFAULT_CONFIG = SystemConfig(
    ocr=OCRConfig(),
    llm=LLMConfig(),
    vision=VisionConfig(),
    processing=ProcessingConfig(),
    analysis=AnalysisConfig()
)


def load_config(config_path: str = None) -> SystemConfig:
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        SystemConfig: 配置对象
    """
    if config_path:
        # 实际实现应该从文件加载配置
        # 这里返回默认配置作为示例
        pass
    
    return DEFAULT_CONFIG


def save_config(config: SystemConfig, config_path: str):
    """
    保存配置到文件
    
    Args:
        config (SystemConfig): 配置对象
        config_path (str): 保存路径
    """
    # 实际实现应该保存配置到文件
    pass


# 常量定义
API_ENDPOINTS = {
    "ocr": "https://api.deepseek.com/v1/ocr",
    "chat": "https://api.deepseek.com/v1/chat/completions",
    "vision": "https://api.deepseek.com/v1/vision/analyze"
}

ERROR_CODES = {
    "INVALID_API_KEY": "E001",
    "RATE_LIMIT_EXCEEDED": "E002", 
    "PROCESSING_TIMEOUT": "E003",
    "INVALID_INPUT_FORMAT": "E004",
    "MODEL_UNAVAILABLE": "E005"
}

# 性能指标标准
PERFORMANCE_BENCHMARKS = {
    "ocr_accuracy_target": 0.95,
    "relation_extraction_precision": 0.85,
    "image_description_accuracy": 0.90,
    "processing_speed_pages_per_minute": 20,
    "token_efficiency_ratio": 0.6
}