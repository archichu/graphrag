"""
工具函数模块
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存查询结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def visualize_graph(graph: nx.Graph, output_path: str = None, max_nodes: int = 100):
    """可视化知识图谱"""
    
    # 检查图是否为空
    if graph.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize")
        return
    
    # 如果节点太多，只显示度数最高的节点
    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        subgraph = graph.subgraph([node for node, _ in top_nodes])
    else:
        subgraph = graph
    
    plt.figure(figsize=(15, 10))
    
    # 计算布局
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    # 绘制节点
    node_sizes = [subgraph.degree(node) * 100 + 100 for node in subgraph.nodes()]
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7)
    
    # 绘制边
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=0.5)
    
    # 绘制标签（只显示重要节点）
    important_nodes = [node for node in subgraph.nodes() if subgraph.degree(node) > 2]
    labels = {node: node[:15] + "..." if len(node) > 15 else node 
             for node in important_nodes}
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    plt.title(f"Knowledge Graph Visualization\n({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graph visualization saved to {output_path}")
    
    plt.show()

def analyze_graph_metrics(graph: nx.Graph) -> Dict[str, Any]:
    """分析图的各种指标"""
    
    metrics = {}
    
    # 基本指标
    metrics['basic'] = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph),
        'num_connected_components': nx.number_connected_components(graph)
    }
    
    # 度分布
    degrees = [d for n, d in graph.degree()]
    if degrees:  # 检查是否有节点
        metrics['degree'] = {
            'average_degree': sum(degrees) / len(degrees),
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'degree_distribution': pd.Series(degrees).describe().to_dict()
        }
    else:
        metrics['degree'] = {
            'average_degree': 0,
            'max_degree': 0,
            'min_degree': 0,
            'degree_distribution': {}
        }
    
    # 中心性指标
    if graph.number_of_nodes() > 0 and graph.number_of_nodes() <= 1000:  # 避免计算过大的图
        try:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
            
            metrics['centrality'] = {
                'top_betweenness': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10],
                'top_closeness': sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10],
                'top_eigenvector': sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        except Exception as e:
            logger.warning(f"Could not compute centrality metrics: {e}")
            metrics['centrality'] = {}
    else:
        metrics['centrality'] = {}
    
    # 聚类系数
    if graph.number_of_nodes() > 0:
        try:
            metrics['clustering'] = {
                'average_clustering': nx.average_clustering(graph),
                'transitivity': nx.transitivity(graph)
            }
        except Exception as e:
            logger.warning(f"Could not compute clustering metrics: {e}")
            metrics['clustering'] = {'average_clustering': 0, 'transitivity': 0}
    else:
        metrics['clustering'] = {'average_clustering': 0, 'transitivity': 0}
    
    return metrics

def plot_degree_distribution(graph: nx.Graph, output_path: str = None):
    """绘制度分布图"""
    degrees = [d for n, d in graph.degree()]
    
    if not degrees:  # 检查是否有节点
        print("Graph is empty, cannot plot degree distribution")
        return
    
    plt.figure(figsize=(12, 4))
    
    # 度分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=min(50, max(degrees) + 1) if degrees else 1, alpha=0.7, edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.grid(True, alpha=0.3)
    
    # 度分布（对数坐标）
    plt.subplot(1, 2, 2)
    degree_counts = pd.Series(degrees).value_counts().sort_index()
    if len(degree_counts) > 0:
        plt.loglog(degree_counts.index, degree_counts.values, 'bo-', alpha=0.7)
    plt.xlabel('Degree (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Degree Distribution (Log-Log)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Degree distribution plot saved to {output_path}")
    
    plt.show()

def create_evaluation_report(results: List[Dict[str, Any]], output_path: str):
    """创建评估报告"""
    
    # 统计信息
    total_queries = len(results)
    successful_queries = len([r for r in results if 'error' not in str(r.get('answer', ''))])
    
    # 置信度分析
    confidences = [r.get('confidence', 0) for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # 实体和社区统计
    total_entities = sum(len(r.get('entities', [])) for r in results)
    total_communities = sum(len(r.get('communities', [])) for r in results)
    
    report = {
        'summary': {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'average_confidence': avg_confidence
        },
        'entity_analysis': {
            'total_entities_retrieved': total_entities,
            'average_entities_per_query': total_entities / total_queries if total_queries > 0 else 0
        },
        'community_analysis': {
            'total_communities_involved': total_communities,
            'average_communities_per_query': total_communities / total_queries if total_queries > 0 else 0
        },
        'detailed_results': results
    }
    
    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Evaluation report saved to {output_path}")
    
    return report

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志"""
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
