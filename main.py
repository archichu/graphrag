"""
GraphRAG项目主程序
演示完整的GraphRAG流程
"""

import asyncio
import os
import argparse
from typing import List, Dict, Any

from src.utils import load_config, save_results, visualize_graph, create_evaluation_report, setup_logging
from src.rag_engine import GraphRAGEngine

async def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='GraphRAG Demo')
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--dataset', default='squad', choices=['squad', 'hotpot_qa', 'natural_questions'], 
                       help='Dataset to use')
    parser.add_argument('--custom-data', help='Path to custom dataset')
    parser.add_argument('--query', help='Single query to process')
    parser.add_argument('--batch-queries', help='File containing queries (one per line)')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate graph visualization')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # 设置日志
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.log_level, os.path.join(args.output_dir, 'graphrag.log'))
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        print("Failed to load config. Using default settings.")
        config = get_default_config()
    
    # 初始化GraphRAG引擎
    print("🚀 Initializing GraphRAG Engine...")
    engine = GraphRAGEngine(config)
    
    try:
        # 从数据集初始化
        await engine.initialize_from_dataset(args.dataset, args.custom_data)
        
        print(f"✅ GraphRAG initialized successfully!")
        print(f"📊 Graph Statistics:")
        stats = engine.get_graph_summary()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # 可视化图（如果请求）
        if args.visualize:
            print("📈 Generating graph visualization...")
            viz_path = os.path.join(args.output_dir, 'graph_visualization.png')
            visualize_graph(engine.graph, viz_path)
        
        # 处理查询
        results = []
        
        if args.query:
            # 单个查询
            print(f"\n🔍 Processing query: {args.query}")
            result = await engine.query(args.query)
            results.append(result)
            
            print(f"💡 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"🏷️  Entities: {[e['text'] for e in result['entities']]}")
            
        elif args.batch_queries:
            # 批量查询
            print(f"\n📝 Processing batch queries from {args.batch_queries}")
            
            with open(args.batch_queries, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = await engine.batch_query(queries)
            
            print(f"✅ Processed {len(results)} queries")
            
        else:
            # 交互式查询
            print("\n💬 Interactive Query Mode (type 'quit' to exit)")
            
            while True:
                query = input("\n🔍 Enter your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                try:
                    result = await engine.query(query)
                    
                    print(f"\n💡 Answer: {result['answer']}")
                    print(f"🎯 Confidence: {result['confidence']:.2f}")
                    
                    if result['entities']:
                        print(f"🏷️  Related Entities: {[e['text'] for e in result['entities'][:5]]}")
                    
                    if result['sources']:
                        print(f"📚 Sources: {len(result['sources'])} document chunks")
                    
                    results.append(result)
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # 保存结果
        if results:
            results_path = os.path.join(args.output_dir, 'query_results.json')
            save_results(results, results_path)
            
            # 创建评估报告
            report_path = os.path.join(args.output_dir, 'evaluation_report.json')
            create_evaluation_report(results, report_path)
            
            print(f"\n📄 Results saved to {results_path}")
            print(f"📊 Evaluation report saved to {report_path}")
        
        # 保存图结构
        graph_path = os.path.join(args.output_dir, 'knowledge_graph.json')
        engine.save_graph(graph_path)
        print(f"🕸️  Knowledge graph saved to {graph_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'model': {
            'llm_model': 'gpt-3.5-turbo',
            'embedding_model': 'text-embedding-ada-002',
            'temperature': 0.1,
            'max_tokens': 2000,
            'local_llm': False,
            'local_embedding_model': 'sentence-transformers/all-MiniLM-L6-v2'
        },
        'graph': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_entities_per_chunk': 10,
            'similarity_threshold': 0.8,
            'community_algorithm': 'leiden',
            'resolution': 1.0
        },
        'retrieval': {
            'top_k_entities': 20,
            'top_k_communities': 5,
            'max_context_length': 8000
        },
        'storage': {
            'working_dir': './graphrag_cache',
            'enable_cache': True
        },
        'dataset': {
            'name': 'squad',
            'max_samples': 1000,
            'split': 'train'
        }
    }

async def demo_queries() -> List[str]:
    """演示查询"""
    return [
        "What is machine learning?",
        "How does neural network work?",
        "What are the applications of artificial intelligence?",
        "Explain the concept of deep learning",
        "What is the difference between supervised and unsupervised learning?"
    ]

if __name__ == "__main__":
    # 设置环境变量（如果需要）
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    asyncio.run(main())
