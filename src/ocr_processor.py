"""
OCR Processor Module
OCR文本处理器模块
基于DeepSeek OCR2实现档案扫描件的文本化处理
"""

import os
import json
from typing import List, Dict, Optional
from PIL import Image
import numpy as np


class DeepSeekOCRProcessor:
    """DeepSeek OCR2处理器类"""
    
    def __init__(self, api_key: str = None):
        """
        初始化OCR处理器
        
        Args:
            api_key (str): DeepSeek API密钥
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model_name = "deepseek-ocr-v2"
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']
        
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        图像预处理
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            Image.Image: 处理后的图像
        """
        try:
            image = Image.open(image_path)
            
            # 旋转校正
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            
            # 亮度和对比度调整
            image = self._adjust_brightness_contrast(image)
            
            return image
        except Exception as e:
            raise Exception(f"图像预处理失败: {str(e)}")
    
    def _adjust_brightness_contrast(self, image: Image.Image, brightness: float = 1.2, 
                                  contrast: float = 1.1) -> Image.Image:
        """调整图像亮度和对比度"""
        # 转换为numpy数组进行处理
        img_array = np.array(image).astype(np.float32)
        
        # 调整亮度
        img_array = img_array * brightness
        
        # 调整对比度
        img_array = (img_array - 128) * contrast + 128
        
        # 限制像素值范围
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def recognize_text(self, image_path: str, language: str = "multi") -> Dict:
        """
        使用DeepSeek OCR2识别文本
        
        Args:
            image_path (str): 图像路径
            language (str): 语言设置 ("en", "zh", "multi")
            
        Returns:
            Dict: 识别结果
        """
        # 预处理图像
        processed_image = self.preprocess_image(image_path)
        
        # 模拟API调用（实际使用时需要替换为真实的API调用）
        result = self._call_deepseek_ocr_api(processed_image, language)
        
        return result
    
    def _call_deepseek_ocr_api(self, image: Image.Image, language: str) -> Dict:
        """
        调用DeepSeek OCR API（模拟实现）
        
        Args:
            image (Image.Image): 处理后的图像
            language (str): 语言设置
            
        Returns:
            Dict: API响应结果
        """
        # 这里应该是实际的API调用代码
        # 示例返回格式：
        return {
            "status": "success",
            "text": "识别的文本内容...",
            "confidence": 0.95,
            "language": language,
            "processing_time": 0.5,
            "tokens_consumed": 100
        }
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     file_extensions: List[str] = None) -> Dict:
        """
        批量处理扫描件
        
        Args:
            input_dir (str): 输入目录
            output_dir (str): 输出目录
            file_extensions (List[str]): 支持的文件扩展名
            
        Returns:
            Dict: 处理统计信息
        """
        if file_extensions is None:
            file_extensions = self.supported_formats
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        processed_files = []
        failed_files = []
        total_tokens = 0
        
        # 遍历输入目录
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, f"{os.path.splitext(relative_path)[0]}.txt")
                    
                    try:
                        # 创建输出子目录
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # 处理文件
                        result = self.recognize_text(input_path)
                        
                        # 保存结果
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result['text'])
                        
                        processed_files.append({
                            'input': input_path,
                            'output': output_path,
                            'confidence': result['confidence'],
                            'tokens': result['tokens_consumed']
                        })
                        
                        total_tokens += result['tokens_consumed']
                        
                    except Exception as e:
                        failed_files.append({
                            'file': input_path,
                            'error': str(e)
                        })
        
        return {
            'processed_count': len(processed_files),
            'failed_count': len(failed_files),
            'total_tokens': total_tokens,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'success_rate': len(processed_files) / (len(processed_files) + len(failed_files)) if processed_files or failed_files else 0
        }


def main():
    """主函数示例"""
    # 初始化处理器
    ocr_processor = DeepSeekOCRProcessor(api_key="your_api_key_here")
    
    # 批量处理示例
    input_directory = "./archive_scans"
    output_directory = "./processed_texts"
    
    results = ocr_processor.batch_process(input_directory, output_directory)
    
    print(f"处理完成！成功: {results['processed_count']}, 失败: {results['failed_count']}")
    print(f"总token消耗: {results['total_tokens']}")
    print(f"成功率: {results['success_rate']:.2%}")


if __name__ == "__main__":
    main()