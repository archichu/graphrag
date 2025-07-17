"""
图构建模块
负责从文档构建知识图谱
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple, Set
import re
from collections import defaultdict, Counter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('graph', {}).get('chunk_size', 1000)
        self.chunk_overlap = config.get('graph', {}).get('chunk_overlap', 200)
        self.max_entities_per_chunk = config.get('graph', {}).get('max_entities_per_chunk', 10)
        self.similarity_threshold = config.get('graph', {}).get('similarity_threshold', 0.8)
        
        # 初始化NLP模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.graph = nx.Graph()
        self.entity_to_chunks = defaultdict(set)
        self.chunk_to_entities = defaultdict(set)
        
    def chunk_text(self, text: str, chunk_id_prefix: str = "") -> List[Dict[str, Any]]:
        """将文本分块"""
        chunks = []
        
        # 简单的句子分割
        sentences = re.split(r'[.!?]+', text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence)
            
            # 如果当前块加上新句子超过限制，保存当前块
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'id': f"{chunk_id_prefix}_chunk_{chunk_id}",
                    'text': current_chunk.strip(),
                    'length': current_length
                })
                
                # 开始新块，保留重叠部分
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence
                current_length += sentence_length + 1
        
        # 添加最后一个块
        if current_chunk.strip():
            chunks.append({
                'id': f"{chunk_id_prefix}_chunk_{chunk_id}",
                'text': current_chunk.strip(),
                'length': current_length
            })
        
        logger.info(f"Created {len(chunks)} chunks for {chunk_id_prefix}")
        return chunks
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体"""
        doc = self.nlp(text)
        
        entities = []
        entity_counts = Counter()
        
        # 提取命名实体
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                entity_text = ent.text.strip()
                if len(entity_text) > 2:  # 过滤太短的实体
                    entity_counts[entity_text] += 1
        
        # 提取名词短语
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # 限制短语长度
                chunk_text = chunk.text.strip()
                if len(chunk_text) > 2:
                    entity_counts[chunk_text] += 1
        
        # 按频次排序，取前N个
        top_entities = entity_counts.most_common(self.max_entities_per_chunk)
        
        for entity_text, count in top_entities:
            entities.append({
                'text': entity_text,
                'type': 'ENTITY',
                'frequency': count,
                'importance': count / len(doc)  # 简单的重要性计算
            })
        
        return entities
    
    def build_entity_relationships(self, chunks: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, Any]]]:
        """构建实体关系"""
        relationships = []
        
        # 收集所有实体
        all_entities = set()
        chunk_entities = {}
        
        for chunk in chunks:
            entities = self.extract_entities(chunk['text'])
            chunk_entities[chunk['id']] = entities
            
            for entity in entities:
                all_entities.add(entity['text'])
                self.entity_to_chunks[entity['text']].add(chunk['id'])
                self.chunk_to_entities[chunk['id']].add(entity['text'])
        
        # 计算实体共现关系
        entity_list = list(all_entities)
        
        for i, entity1 in enumerate(entity_list):
            for j, entity2 in enumerate(entity_list[i+1:], i+1):
                # 计算共现频次
                common_chunks = self.entity_to_chunks[entity1] & self.entity_to_chunks[entity2]
                
                if len(common_chunks) > 0:
                    # 计算关系强度
                    strength = len(common_chunks) / min(len(self.entity_to_chunks[entity1]), 
                                                      len(self.entity_to_chunks[entity2]))
                    
                    if strength >= 0.1:  # 最小关系强度阈值
                        relationships.append((
                            entity1,
                            entity2,
                            {
                                'weight': strength,
                                'co_occurrence': len(common_chunks),
                                'type': 'CO_OCCURRENCE'
                            }
                        ))
        
        logger.info(f"Built {len(relationships)} entity relationships")
        return relationships
    
    def build_graph_from_documents(self, documents: List[Dict[str, str]]) -> nx.Graph:
        """从文档构建知识图谱"""
        logger.info("Building knowledge graph from documents...")
        
        all_chunks = []
        
        # 处理每个文档
        for doc in documents:
            doc_chunks = self.chunk_text(doc['content'], doc['id'])
            
            # 为每个块添加文档信息
            for chunk in doc_chunks:
                chunk['document_id'] = doc['id']
                chunk['document_title'] = doc['title']
                chunk['source'] = doc['source']
            
            all_chunks.extend(doc_chunks)
        
        # 构建实体关系
        relationships = self.build_entity_relationships(all_chunks)
        
        # 添加节点和边到图中
        for entity1, entity2, attrs in relationships:
            self.graph.add_edge(entity1, entity2, **attrs)
        
        # 添加节点属性
        for node in self.graph.nodes():
            chunks_with_entity = list(self.entity_to_chunks[node])
            self.graph.nodes[node]['chunks'] = chunks_with_entity
            self.graph.nodes[node]['frequency'] = len(chunks_with_entity)
        
        # 存储块信息
        self.chunks = {chunk['id']: chunk for chunk in all_chunks}
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def detect_communities(self) -> Dict[str, int]:
        """检测社区结构"""
        try:
            import networkx.algorithms.community as nx_comm
            
            # 使用Louvain算法检测社区
            communities = nx_comm.louvain_communities(self.graph, seed=42)
            
            # 创建节点到社区的映射
            node_to_community = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_to_community[node] = i
            
            logger.info(f"Detected {len(communities)} communities")
            return node_to_community
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
        }
        
        if stats['num_nodes'] > 0:
            stats['average_degree'] = sum(dict(self.graph.degree()).values()) / stats['num_nodes']
        
        return stats
