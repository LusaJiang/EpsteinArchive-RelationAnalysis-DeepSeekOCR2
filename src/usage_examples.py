"""
Usage Examples
使用示例
演示如何使用Epstein档案分析系统的各个模块
"""

import os
import json
from pathlib import Path

# 导入系统模块
from .ocr_processor import DeepSeekOCRProcessor
from .llm_analyzer import DeepSeekLLMAnalyzer  
from .vision_analyzer import DeepSeekVisionAnalyzer
from .main_pipeline import ArchiveAnalysisPipeline
from .config import DEFAULT_CONFIG, load_config


def example_ocr_processing():
    """OCR处理示例"""
    print("=== OCR文本化处理示例 ===")
    
    # 初始化OCR处理器
    ocr_processor = DeepSeekOCRProcessor(api_key="your_api_key_here")
    
    # 单文件处理示例
    sample_image = "./samples/epstein_document.jpg"
    if os.path.exists(sample_image):
        try:
            result = ocr_processor.recognize_text(sample_image, language="multi")
            print(f"识别结果: {result['text'][:100]}...")
            print(f"置信度: {result['confidence']}")
            print(f"Token消耗: {result['tokens_consumed']}")
        except Exception as e:
            print(f"处理失败: {str(e)}")
    
    # 批量处理示例
    input_dir = "./archive_scans"
    output_dir = "./processed_texts"
    
    if os.path.exists(input_dir):
        stats = ocr_processor.batch_process(input_dir, output_dir)
        print(f"批量处理统计: {stats}")


def example_text_analysis():
    """文本分析示例"""
    print("\n=== 文本关系抽取示例 ===")
    
    # 初始化分析器
    analyzer = DeepSeekLLMAnalyzer(api_key="your_api_key_here")
    
    # 示例文本
    sample_text = """
    Jeffrey Epstein invited Donald Trump to Lolita Island for a private meeting in March 2018.
    The discussion involved business arrangements and travel plans to Palm Beach.
    Bill Clinton was also mentioned as having previous visits to the location.
    Epstein coordinated with Ghislaine Maxwell to arrange the logistics.
    """
    
    # 分析文本
    result = analyzer.analyze_document(sample_text)
    
    print("识别的实体:")
    for entity in result['entities']:
        print(f"  - {entity['text']} ({entity['type']})")
    
    print("\n抽取的关系:")
    for relation in result['relations']:
        print(f"  - {relation['source']} --{relation['type']}--> {relation['target']}")
    
    print(f"\n统计信息: {result['statistics']}")


def example_vision_analysis():
    """视觉分析示例"""
    print("\n=== 图片分析示例 ===")
    
    # 初始化视觉分析器
    vision_analyzer = DeepSeekVisionAnalyzer(api_key="your_api_key_here")
    
    # 示例图片
    sample_image = "./samples/archive_photo.jpg"
    context_text = "2018年3月爱泼斯坦在萝莉岛接待安德鲁王子"
    
    if os.path.exists(sample_image):
        try:
            # 综合分析
            result = vision_analyzer.analyze_with_context(sample_image, context_text)
            
            print("场景描述:")
            print(f"  {result['scene_description']['description']}")
            print(f"  设置: {result['scene_description']['setting']}")
            print(f"  时期: {result['scene_description']['time_period']}")
            
            print("\n识别要素:")
            for element in result['identified_elements']:
                print(f"  {element['type']}: {element['description']}")
            
            print("\n人物识别:")
            for identification in result['person_identifications']:
                print(f"  {identification['detected_person']} -> {identification['matched_person']}")
                
        except Exception as e:
            print(f"分析失败: {str(e)}")


def example_complete_pipeline():
    """完整流水线示例"""
    print("\n=== 完整分析流水线示例 ===")
    
    # 初始化流水线
    pipeline = ArchiveAnalysisPipeline(
        api_key="your_api_key_here",
        output_dir="./analysis_output"
    )
    
    # 处理档案
    archive_path = "./epstein_archive"
    
    if os.path.exists(archive_path):
        try:
            results = pipeline.process_archive(archive_path)
            
            print("分析完成!")
            print(f"处理文档: {results['pipeline_statistics']['processed_documents']}")
            print(f"处理图片: {results['pipeline_statistics']['processed_images']}")
            print(f"识别实体: {results['pipeline_statistics']['total_entities']}")
            print(f"抽取关系: {results['pipeline_statistics']['total_relations']}")
            
        except Exception as e:
            print(f"流水线执行失败: {str(e)}")
    else:
        print(f"档案目录不存在: {archive_path}")


def example_configuration():
    """配置使用示例"""
    print("\n=== 配置使用示例 ===")
    
    # 使用默认配置
    config = DEFAULT_CONFIG
    
    print("OCR配置:")
    print(f"  模型: {config.ocr.model_name}")
    print(f"  语言: {config.ocr.language}")
    print(f"  置信度阈值: {config.ocr.confidence_threshold}")
    
    print("\nLLM配置:")
    print(f"  模型: {config.llm.model_name}")
    print(f"  最大token: {config.llm.max_tokens}")
    print(f"  温度: {config.llm.temperature}")
    
    print("\n视觉配置:")
    print(f"  模型: {config.vision.model_name}")
    print(f"  描述详细程度: {config.vision.description_detail_level}")


def example_custom_analysis():
    """自定义分析示例"""
    print("\n=== 自定义分析示例 ===")
    
    # 创建自定义分析流程
    def custom_person_network_analysis(text_files: List[str], image_files: List[str]):
        """自定义人物网络分析"""
        
        # 初始化组件
        ocr_processor = DeepSeekOCRProcessor()
        llm_analyzer = DeepSeekLLMAnalyzer()
        vision_analyzer = DeepSeekVisionAnalyzer()
        
        # 存储所有结果
        all_entities = []
        all_relations = []
        person_images = {}
        
        # 处理文本文件
        for text_file in text_files:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = llm_analyzer.analyze_document(text)
            all_entities.extend(result['entities'])
            all_relations.extend(result['relations'])
        
        # 处理图片文件
        for image_file in image_files:
            try:
                elements = vision_analyzer.identify_elements(image_file)
                persons = [e for e in elements if e.element_type == "PERSON"]
                
                for person in persons:
                    if person.description not in person_images:
                        person_images[person.description] = []
                    person_images[person.description].append({
                        'image': image_file,
                        'confidence': person.confidence
                    })
            except Exception as e:
                print(f"处理图片 {image_file} 失败: {str(e)}")
        
        # 构建人物网络
        person_network = build_person_network(all_entities, all_relations, person_images)
        
        return person_network
    
    def build_person_network(entities, relations, person_images):
        """构建人物网络"""
        network = {
            'nodes': {},
            'edges': [],
            'image_evidence': person_images
        }
        
        # 添加节点
        for entity in entities:
            if entity['type'] == 'PERSON':
                person_name = entity['text']
                if person_name not in network['nodes']:
                    network['nodes'][person_name] = {
                        'entity_count': 0,
                        'confidence_sum': 0
                    }
                network['nodes'][person_name]['entity_count'] += 1
                network['nodes'][person_name]['confidence_sum'] += entity['confidence']
        
        # 添加边
        for relation in relations:
            network['edges'].append({
                'source': relation['source'],
                'target': relation['target'],
                'type': relation['type'],
                'weight': relation['confidence']
            })
        
        return network
    
    # 使用示例
    text_files = ["./texts/doc1.txt", "./texts/doc2.txt"]
    image_files = ["./images/img1.jpg", "./images/img2.jpg"]
    
    # 过滤存在的文件
    existing_texts = [f for f in text_files if os.path.exists(f)]
    existing_images = [f for f in image_files if os.path.exists(f)]
    
    if existing_texts and existing_images:
        network = custom_person_network_analysis(existing_texts, existing_images)
        print("自定义人物网络分析完成")
        print(f"节点数: {len(network['nodes'])}")
        print(f"边数: {len(network['edges'])}")
        print(f"有图像证据的人物: {len(network['image_evidence'])}")


def main():
    """运行所有示例"""
    print("Epstein档案分析系统使用示例")
    print("=" * 50)
    
    # 运行各个示例
    example_configuration()
    example_ocr_processing()
    example_text_analysis()
    example_vision_analysis()
    example_complete_pipeline()
    example_custom_analysis()
    
    print("\n所有示例运行完成!")


if __name__ == "__main__":
    main()