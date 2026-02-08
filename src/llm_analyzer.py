"""
LLM Text Analyzer Module
大语言模型文本分析模块
基于DeepSeek V3.2实现文本分析和人物关系抽取
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RelationType(Enum):
    """人物关系类型枚举"""
    SOCIAL_INTERACTION = "social_interaction"  # 社交往来
    BUSINESS_ASSOCIATION = "business_association"  # 商业关联
    POLITICAL_CONNECTION = "political_connection"  # 政治联系
    FAMILY_RELATIONSHIP = "family_relationship"  # 家庭关系
    TRAVEL_COMPANION = "travel_companion"  # 行程同伴
    FINANCIAL_TIE = "financial_tie"  # 财务关联


@dataclass
class Entity:
    """实体类"""
    text: str
    entity_type: str  # PERSON, LOCATION, ORGANIZATION, EVENT
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class Relation:
    """关系类"""
    source_entity: str
    target_entity: str
    relation_type: RelationType
    evidence_text: str
    confidence: float
    context: str


class DeepSeekLLMAnalyzer:
    """DeepSeek大语言模型分析器"""
    
    def __init__(self, api_key: str = None, model_name: str = "deepseek-chat"):
        """
        初始化分析器
        
        Args:
            api_key (str): API密钥
            model_name (str): 模型名称
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.model_name = model_name
        self.person_keywords = [
            'epstein', 'jeffrey', 'trump', 'donald', 'clinton', 'bill',
            'gates', 'bill', 'musk', 'elon', 'prince', 'andrew',
            'maxwell', 'ghislaine', 'obama', 'barack'
        ]
        
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 处理后的文本
        """
        # 转换为小写
        text = text.lower()
        
        # 移除多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        
        return text
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        命名实体识别
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[Entity]: 实体列表
        """
        entities = []
        processed_text = self.preprocess_text(text)
        
        # 基于规则的实体识别
        person_patterns = [
            r'\b(jeffrey\s+epstein)\b',
            r'\b(donald\s+trump)\b',
            r'\b(bill\s+clinton)\b',
            r'\b(elon\s+musk)\b',
            r'\b(andrew\s+prince)\b',
            r'\b(ghislaine\s+maxwell)\b',
            r'\b(barack\s+obama)\b',
        ]
        
        location_patterns = [
            r'\b(lolita\s+island)\b',
            r'\b(palm\s+beach)\b',
            r'\b(new\s+york)\b',
            r'\b(washington)\b',
        ]
        
        # 识别人物实体
        for pattern in person_patterns:
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type="PERSON",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9
                ))
        
        # 识别地点实体
        for pattern in location_patterns:
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    entity_type="LOCATION",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.85
                ))
        
        # 调用LLM进行更精确的实体识别（模拟）
        llm_entities = self._call_llm_entity_recognition(processed_text)
        entities.extend(llm_entities)
        
        return entities
    
    def _call_llm_entity_recognition(self, text: str) -> List[Entity]:
        """
        调用LLM进行实体识别（模拟实现）
        
        Args:
            text (str): 输入文本
            
        Returns:
            List[Entity]: 实体列表
        """
        # 实际实现应该调用DeepSeek API
        return []  # 返回空列表作为示例
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        关系抽取
        
        Args:
            text (str): 输入文本
            entities (List[Entity]): 实体列表
            
        Returns:
            List[Relation]: 关系列表
        """
        relations = []
        processed_text = self.preprocess_text(text)
        
        # 基于关键词的关系抽取规则
        relation_patterns = [
            # 社交往来关系
            (r'(invited\s+\w+\s+to|met\s+with|discussed\s+plans\s+with)', 
             RelationType.SOCIAL_INTERACTION),
            # 商业关联
            (r'(business\s+deal|financial\s+arrangement|investment\s+from)', 
             RelationType.BUSINESS_ASSOCIATION),
            # 行程同伴
            (r'(traveled\s+together|accompanied\s+by|joint\s+trip\s+to)', 
             RelationType.TRAVEL_COMPANION),
            # 财务关联
            (r'(received\s+payment|financial\s+support|money\s+transfer)', 
             RelationType.FINANCIAL_TIE),
        ]
        
        # 提取人物实体
        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        
        # 基于模式匹配的关系抽取
        for pattern, rel_type in relation_patterns:
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            for match in matches:
                # 查找附近的实体
                nearby_entities = self._find_nearby_entities(
                    match.start(), match.end(), person_entities, window=100
                )
                
                if len(nearby_entities) >= 2:
                    relations.append(Relation(
                        source_entity=nearby_entities[0].text,
                        target_entity=nearby_entities[1].text,
                        relation_type=rel_type,
                        evidence_text=match.group(),
                        confidence=0.8,
                        context=self._extract_context(processed_text, match.start(), match.end())
                    ))
        
        # 调用LLM进行深度关系抽取
        llm_relations = self._call_llm_relation_extraction(processed_text, entities)
        relations.extend(llm_relations)
        
        return relations
    
    def _find_nearby_entities(self, start_pos: int, end_pos: int, 
                            entities: List[Entity], window: int = 50) -> List[Entity]:
        """查找指定位置附近的实体"""
        nearby = []
        for entity in entities:
            if (abs(entity.start_pos - start_pos) <= window or 
                abs(entity.start_pos - end_pos) <= window):
                nearby.append(entity)
        return nearby[:3]  # 最多返回3个实体
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, 
                        context_length: int = 100) -> str:
        """提取上下文文本"""
        start = max(0, start_pos - context_length)
        end = min(len(text), end_pos + context_length)
        return text[start:end]
    
    def _call_llm_relation_extraction(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        调用LLM进行关系抽取（模拟实现）
        
        Args:
            text (str): 输入文本
            entities (List[Entity]): 实体列表
            
        Returns:
            List[Relation]: 关系列表
        """
        # 实际实现应该调用DeepSeek API进行深度语义分析
        return []  # 返回空列表作为示例
    
    def analyze_document(self, text: str) -> Dict:
        """
        分析单个文档
        
        Args:
            text (str): 文档文本
            
        Returns:
            Dict: 分析结果
        """
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 实体识别
        entities = self.extract_entities(processed_text)
        
        # 关系抽取
        relations = self.extract_relations(processed_text, entities)
        
        # 统计信息
        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        location_entities = [e for e in entities if e.entity_type == "LOCATION"]
        
        return {
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type,
                    "confidence": e.confidence
                } for e in entities
            ],
            "relations": [
                {
                    "source": r.source_entity,
                    "target": r.target_entity,
                    "type": r.relation_type.value,
                    "confidence": r.confidence,
                    "evidence": r.evidence_text
                } for r in relations
            ],
            "statistics": {
                "total_entities": len(entities),
                "person_entities": len(person_entities),
                "location_entities": len(location_entities),
                "total_relations": len(relations)
            }
        }
    
    def batch_analyze(self, text_files: List[str]) -> Dict:
        """
        批量分析文档
        
        Args:
            text_files (List[str]): 文本文件路径列表
            
        Returns:
            Dict: 批量分析结果
        """
        all_results = []
        total_entities = 0
        total_relations = 0
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = self.analyze_document(text)
                result['file_path'] = file_path
                all_results.append(result)
                
                total_entities += result['statistics']['total_entities']
                total_relations += result['statistics']['total_relations']
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        return {
            "results": all_results,
            "summary": {
                "processed_files": len(all_results),
                "total_entities": total_entities,
                "total_relations": total_relations,
                "avg_entities_per_doc": total_entities / len(all_results) if all_results else 0,
                "avg_relations_per_doc": total_relations / len(all_results) if all_results else 0
            }
        }


def main():
    """主函数示例"""
    # 初始化分析器
    analyzer = DeepSeekLLMAnalyzer(api_key="your_api_key_here")
    
    # 分析单个文档示例
    sample_text = """
    Jeffrey Epstein invited Donald Trump to Lolita Island for a private meeting.
    The discussion involved business arrangements and travel plans to Palm Beach.
    Bill Clinton was also mentioned as having previous visits to the location.
    """
    
    result = analyzer.analyze_document(sample_text)
    
    print("实体识别结果:")
    for entity in result['entities']:
        print(f"  {entity['text']} ({entity['type']}) - 置信度: {entity['confidence']}")
    
    print("\n关系抽取结果:")
    for relation in result['relations']:
        print(f"  {relation['source']} --{relation['type']}--> {relation['target']}")
    
    print(f"\n统计信息: {result['statistics']}")


if __name__ == "__main__":
    main()