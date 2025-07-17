"""
RAG引擎模块
整合图检索和生成功能
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import numpy as np
from collections import defaultdict
import logging

# 导入nano-graphrag
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import compute_args_hash

from .graph_builder import GraphBuilder
from .data_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGEngine:
    """GraphRAG引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.working_dir = config.get('storage', {}).get('working_dir', './graphrag_cache')
        self.enable_cache = config.get('storage', {}).get('enable_cache', True)
        
        # 设置 API 配置
        api_key = config.get('model', {}).get('api_key')
        base_url = config.get('model', {}).get('base_url')
        
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        if base_url:
            os.environ['OPENAI_BASE_URL'] = base_url
        
        # 初始化组件
        self.data_loader = DatasetLoader(config)
        self.graph_builder = GraphBuilder(config)
        
        # 初始化nano-graphrag
        if GraphRAG is not None:
            try:
                self.rag = GraphRAG(working_dir=self.working_dir)
                logger.info("GraphRAG initialized successfully")
            except Exception as e:
                logger.warning(f"GraphRAG初始化失败: {e}")
                self.rag = None
        else:
            self.rag = None
            logger.warning("GraphRAG未安装，使用简化模式")
        
        self.documents = []
        self.graph = None
        self.communities = {}

        
    async def initialize_from_dataset(self, dataset_name: Optional[str] = None, custom_path: Optional[str] = None):
        """从数据集初始化"""
        logger.info("Initializing GraphRAG from dataset...")
        
        # 加载数据
        if dataset_name:
            self.config['dataset']['name'] = dataset_name
            self.data_loader = DatasetLoader(self.config)
        
        self.documents = self.data_loader.load_dataset(custom_path)
        
        if not self.documents:
            raise ValueError("No documents loaded")
        
        # 构建图
        self.graph = self.graph_builder.build_graph_from_documents(self.documents)
        self.communities = self.graph_builder.detect_communities()
        
        # 插入文档到nano-graphrag
        logger.info("Inserting documents into nano-graphrag...")
        for doc in self.documents:
            content = f"Title: {doc['title']}\n\nContent: {doc['content']}"
            await self.rag.ainsert(content)
        
        logger.info("GraphRAG initialization completed")
    
    async def query(self, question: str, mode: str = "global") -> Dict[str, Any]:
        """查询接口"""
        logger.info(f"Processing query: {question}")
        
        try:
            # 使用nano-graphrag进行查询
            if mode == "global":
                result = await self.rag.aquery(question, param=QueryParam(mode="global"))
            else:
                result = await self.rag.aquery(question, param=QueryParam(mode="local"))
            
            # 增强结果with图信息
            enhanced_result = await self._enhance_with_graph_info(question, result)
            
            return {
                'question': question,
                'answer': enhanced_result['answer'],
                'mode': mode,
                'sources': enhanced_result.get('sources', []),
                'entities': enhanced_result.get('entities', []),
                'communities': enhanced_result.get('communities', []),
                'confidence': enhanced_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                'question': question,
                'answer': f"Error processing query: {str(e)}",
                'mode': mode,
                'sources': [],
                'entities': [],
                'communities': [],
                'confidence': 0.0
            }
    
    async def _enhance_with_graph_info(self, question: str, base_result: str) -> Dict[str, Any]:
        """使用图信息增强结果"""
        
        # 提取问题中的实体
        question_entities = self.graph_builder.extract_entities(question)
        relevant_entities = []
        relevant_communities = set()
        
        for entity_info in question_entities:
            entity_text = entity_info['text']
            if entity_text in self.graph.nodes():
                relevant_entities.append({
                    'text': entity_text,
                    'frequency': self.graph.nodes[entity_text].get('frequency', 0),
                    'community': self.communities.get(entity_text, -1)
                })
                
                if entity_text in self.communities:
                    relevant_communities.add(self.communities[entity_text])
        
        # 获取相关文档块
        relevant_chunks = set()
        for entity_info in relevant_entities:
            entity_text = entity_info['text']
            if entity_text in self.graph_builder.entity_to_chunks:
                relevant_chunks.update(self.graph_builder.entity_to_chunks[entity_text])
        
        # 构建来源信息
        sources = []
        for chunk_id in list(relevant_chunks)[:5]:  # 限制来源数量
            if chunk_id in self.graph_builder.chunks:
                chunk = self.graph_builder.chunks[chunk_id]
                sources.append({
                    'chunk_id': chunk_id,
                    'document_title': chunk.get('document_title', ''),
                    'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                })
        
        return {
            'answer': base_result,
            'entities': relevant_entities,
            'communities': list(relevant_communities),
            'sources': sources,
            'confidence': min(1.0, len(relevant_entities) * 0.2 + 0.3)  # 简单的置信度计算
        }
    
    def get_entity_info(self, entity: str) -> Dict[str, Any]:
        """获取实体信息"""
        if entity not in self.graph.nodes():
            return {'error': f'Entity "{entity}" not found in graph'}
        
        node_data = self.graph.nodes[entity]
        neighbors = list(self.graph.neighbors(entity))
        
        return {
            'entity': entity,
            'frequency': node_data.get('frequency', 0),
            'community': self.communities.get(entity, -1),
            'neighbors': neighbors[:10],  # 限制邻居数量
            'chunks': list(node_data.get('chunks', []))[:5],  # 限制块数量
            'degree': self.graph.degree(entity)
        }
    
    def get_community_info(self, community_id: int) -> Dict[str, Any]:
        """获取社区信息"""
        community_entities = [entity for entity, comm_id in self.communities.items() 
                            if comm_id == community_id]
        
        if not community_entities:
            return {'error': f'Community {community_id} not found'}
        
        # 获取社区内的边
        subgraph = self.graph.subgraph(community_entities)
        
        return {
            'community_id': community_id,
            'entities': community_entities,
            'num_entities': len(community_entities),
            'num_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph) if len(community_entities) > 1 else 0.0
        }
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """获取图摘要"""
        stats = self.graph_builder.get_graph_statistics()
        
        # 添加社区信息
        community_sizes = defaultdict(int)
        for entity, comm_id in self.communities.items():
            community_sizes[comm_id] += 1
        
        stats['num_communities'] = len(community_sizes)
        stats['largest_community_size'] = max(community_sizes.values()) if community_sizes else 0
        stats['average_community_size'] = np.mean(list(community_sizes.values())) if community_sizes else 0
        
        return stats
    
    async def batch_query(self, questions: List[str], mode: str = "global") -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        
        for question in questions:
            result = await self.query(question, mode)
            results.append(result)
            
            # 添加延迟避免API限制
            await asyncio.sleep(0.1)
        
        return results
    
    def save_graph(self, filepath: str):
        """保存图结构"""
        graph_data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True)),
            'communities': self.communities,
            'chunks': self.graph_builder.chunks
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Graph saved to {filepath}")
    
    def load_graph(self, filepath: str):
        """加载图结构"""
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # 重建图
        self.graph = nx.Graph()
        
        # 添加节点
        for node, data in graph_data['nodes'].items():
            self.graph.add_node(node, **data)
        
        # 添加边
        for edge_data in graph_data['edges']:
            if len(edge_data) == 3:
                u, v, attrs = edge_data
                self.graph.add_edge(u, v, **attrs)
            else:
                u, v = edge_data
                self.graph.add_edge(u, v)
        
        self.communities = graph_data.get('communities', {})
        self.graph_builder.chunks = graph_data.get('chunks', {})
        
        logger.info(f"Graph loaded from {filepath}")
