"""
数据加载模块
支持多种开源数据集的加载和预处理
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """数据集加载器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get('dataset', {}).get('name', 'squad')
        self.max_samples = config.get('dataset', {}).get('max_samples', 1000)
        self.split = config.get('dataset', {}).get('split', 'train')
        
    def _try_load_with_mirror(self, dataset_name: str, split: str):
        """尝试使用不同的源加载数据集"""
        
        # 方法1: 直接从 Hugging Face Hub 加载
        try:
            logger.info("尝试从 Hugging Face Hub 加载数据集...")
            from datasets import load_dataset
            
            # 清除可能的环境变量
            if 'HF_ENDPOINT' in os.environ:
                del os.environ['HF_ENDPOINT']
            
            dataset = load_dataset(dataset_name, split=split)
            logger.info("✅ 成功从 Hugging Face Hub 加载数据集")
            return dataset
            
        except Exception as e:
            logger.warning(f"从 Hugging Face Hub 加载失败: {e}")
            
            # 方法2: 使用 HF Mirror
            try:
                logger.info("尝试从 HF Mirror 加载数据集...")
                
                # 设置 HF Mirror 环境变量
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                
                # 重新导入以应用新的环境变量
                import importlib
                import datasets
                importlib.reload(datasets)
                
                from datasets import load_dataset
                dataset = load_dataset(dataset_name, split=split)
                logger.info("✅ 成功从 HF Mirror 加载数据集")
                return dataset
                
            except Exception as e2:
                logger.warning(f"从 HF Mirror 加载失败: {e2}")
                
                # 方法3: 尝试其他镜像
                try:
                    logger.info("尝试从其他镜像加载数据集...")
                    os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
                    
                    import importlib
                    import datasets
                    importlib.reload(datasets)
                    
                    from datasets import load_dataset
                    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                    logger.info("✅ 成功从备用镜像加载数据集")
                    return dataset
                    
                except Exception as e3:
                    logger.error(f"所有数据源都加载失败: {e3}")
                    return None
    
    def load_squad_dataset(self) -> List[Dict[str, str]]:
        """加载SQuAD数据集"""
        logger.info("Loading SQuAD dataset...")
        
        # 尝试加载数据集
        dataset = self._try_load_with_mirror("squad", self.split)
        
        if dataset is not None:
            try:
                documents = []
                seen_contexts = set()
                
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                        
                    context = item['context']
                    title = item['title']
                    
                    # 避免重复文档
                    if context not in seen_contexts:
                        documents.append({
                            'id': f"squad_{i}",
                            'title': title,
                            'content': context,
                            'source': 'SQuAD',
                            'questions': [item['question']],
                            'answers': [item['answers']['text'][0] if item['answers']['text'] else ""]
                        })
                        seen_contexts.add(context)
                        
                logger.info(f"Loaded {len(documents)} unique documents from SQuAD")
                return documents
                
            except Exception as e:
                logger.error(f"Error processing SQuAD dataset: {e}")
        
        # 如果所有方法都失败，使用示例数据
        logger.info("使用示例数据替代...")
        return self._get_sample_documents()
    
    def load_hotpot_qa_dataset(self) -> List[Dict[str, str]]:
        """加载HotpotQA数据集"""
        logger.info("Loading HotpotQA dataset...")
        
        # 尝试加载数据集
        dataset = self._try_load_with_mirror("hotpot_qa", f"{self.split}[:100]")  # 限制数量避免太大
        
        if dataset is not None:
            try:
                documents = []
                
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                    
                    # 合并所有支持文档
                    context_parts = []
                    for title, sentences in item['context']:
                        context_parts.append(f"**{title}**\n" + " ".join(sentences))
                    
                    full_context = "\n\n".join(context_parts)
                    
                    documents.append({
                        'id': f"hotpot_{i}",
                        'title': f"Multi-hop Question {i}",
                        'content': full_context,
                        'source': 'HotpotQA',
                        'questions': [item['question']],
                        'answers': [item['answer']]
                    })
                    
                logger.info(f"Loaded {len(documents)} documents from HotpotQA")
                return documents
                
            except Exception as e:
                logger.error(f"Error processing HotpotQA dataset: {e}")
        
        # 如果失败，使用示例数据
        logger.info("使用示例数据替代...")
        return self._get_sample_documents()
    
    def load_natural_questions_dataset(self) -> List[Dict[str, str]]:
        """加载Natural Questions数据集"""
        logger.info("Loading Natural Questions dataset...")
        
        # 尝试加载数据集
        dataset = self._try_load_with_mirror("natural_questions", f"{self.split}[:100]")
        
        if dataset is not None:
            try:
                documents = []
                
                for i, item in enumerate(dataset):
                    if i >= self.max_samples:
                        break
                    
                    # 提取文档内容
                    document_text = item['document']['tokens']['token']
                    content = " ".join([token for token in document_text if token.strip()])
                    
                    # 提取问题
                    question = item['question']['text']
                    
                    documents.append({
                        'id': f"nq_{i}",
                        'title': item['document']['title'] or f"Document {i}",
                        'content': content[:5000],  # 限制长度
                        'source': 'Natural Questions',
                        'questions': [question],
                        'answers': [""]  # NQ的答案提取比较复杂，这里简化
                    })
                    
                logger.info(f"Loaded {len(documents)} documents from Natural Questions")
                return documents
                
            except Exception as e:
                logger.error(f"Error processing Natural Questions dataset: {e}")
        
        # 如果失败，使用示例数据
        logger.info("使用示例数据替代...")
        return self._get_sample_documents()
    
    def _get_sample_documents(self) -> List[Dict[str, str]]:
        """获取示例文档（当数据集加载失败时）"""
        logger.info("使用内置示例数据...")
        
        return [
            {
                'id': 'sample_1',
                'title': 'Machine Learning Fundamentals',
                'content': 'Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build a mathematical model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.',
                'source': 'Sample',
                'questions': ['What is machine learning?', 'How do machine learning algorithms work?'],
                'answers': ['A method of data analysis that automates analytical model building', 'They build mathematical models based on training data']
            },
            {
                'id': 'sample_2', 
                'title': 'Artificial Intelligence Overview',
                'content': 'Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. AI research has been highly successful in developing effective techniques for solving a wide range of problems, from game playing to medical diagnosis.',
                'source': 'Sample',
                'questions': ['What is artificial intelligence?', 'What are the characteristics of AI?'],
                'answers': ['The simulation of human intelligence in machines', 'Ability to rationalize and take goal-oriented actions']
            },
            {
                'id': 'sample_3',
                'title': 'Deep Learning and Neural Networks',
                'content': 'Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics and drug design. A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.',
                'source': 'Sample',
                'questions': ['What is deep learning?', 'What are neural networks?'],
                'answers': ['Part of machine learning methods based on artificial neural networks', 'Networks of artificial neurons or nodes']
            },
            {
                'id': 'sample_4',
                'title': 'Natural Language Processing',
                'content': 'Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.',
                'source': 'Sample',
                'questions': ['What is natural language processing?', 'What are the challenges in NLP?'],
                'answers': ['A field concerned with interactions between computers and human language', 'Speech recognition, natural language understanding, and generation']
            },
            {
                'id': 'sample_5',
                'title': 'Computer Vision and Image Recognition',
                'content': 'Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do. Computer vision tasks include methods for acquiring, processing, analyzing and understanding digital images, and extraction of high-dimensional data from the real world in order to produce numerical or symbolic information. Image recognition is the ability of a machine or software to identify objects, places, people, writing and actions in images.',
                'source': 'Sample',
                'questions': ['What is computer vision?', 'What is image recognition?'],
                'answers': ['A field dealing with how computers understand digital images', 'The ability to identify objects and actions in images']
            },
            {
                'id': 'sample_6',
                'title': 'Data Science and Analytics',
                'content': 'Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from many structural and unstructured data. Data science is related to data mining, machine learning and big data. Data science is a concept to unify statistics, data analysis, informatics, and their related methods in order to understand and analyze actual phenomena with data. It uses techniques and theories drawn from many fields within the context of mathematics, statistics, computer science, information science, and domain knowledge.',
                'source': 'Sample',
                'questions': ['What is data science?', 'How does data science relate to other fields?'],
                'answers': ['An interdisciplinary field that extracts knowledge from data', 'It combines statistics, computer science, and domain knowledge']
            },
            {
                'id': 'sample_7',
                'title': 'Robotics and Automation',
                'content': 'Robotics is an interdisciplinary field that integrates computer science and engineering. Robotics involves design, construction, operation, and use of robots, as well as computer systems for their control, sensory feedback, and information processing. These technologies are used to develop machines that can substitute for humans and replicate human actions. Robots can be used in many situations and for lots of purposes, but today many are used in dangerous environments, manufacturing processes, or where humans cannot survive.',
                'source': 'Sample',
                'questions': ['What is robotics?', 'Where are robots used?'],
                'answers': ['An interdisciplinary field integrating computer science and engineering', 'In dangerous environments, manufacturing, and places humans cannot survive']
            },
            {
                'id': 'sample_8',
                'title': 'Cloud Computing and Distributed Systems',
                'content': 'Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations, each location being a data center. Cloud computing relies on sharing of resources to achieve coherence and typically using a pay-as-you-go model which can help in reducing capital expenses but may also lead to unexpected operating expenses for unaware users.',
                'source': 'Sample',
                'questions': ['What is cloud computing?', 'How does cloud computing work?'],
                'answers': ['On-demand availability of computer resources without direct management', 'By sharing resources across distributed data centers']
            }
        ]
    
    def load_custom_dataset(self, file_path: str) -> List[Dict[str, str]]:
        """加载自定义数据集"""
        logger.info(f"Loading custom dataset from {file_path}")
        
        documents = []
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for i, item in enumerate(data):
                    documents.append({
                        'id': item.get('id', f"custom_{i}"),
                        'title': item.get('title', f"Document {i}"),
                        'content': item.get('content', ''),
                        'source': 'Custom',
                        'questions': item.get('questions', []),
                        'answers': item.get('answers', [])
                    })
                    
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                for i, row in df.iterrows():
                    documents.append({
                        'id': row.get('id', f"custom_{i}"),
                        'title': row.get('title', f"Document {i}"),
                        'content': row.get('content', ''),
                        'source': 'Custom',
                        'questions': [row.get('question', '')] if 'question' in row else [],
                        'answers': [row.get('answer', '')] if 'answer' in row else []
                    })
                    
            logger.info(f"Loaded {len(documents)} documents from custom dataset")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            return []
    
    def load_dataset(self, custom_path: Optional[str] = None) -> List[Dict[str, str]]:
        """根据配置加载数据集"""
        
        if custom_path:
            return self.load_custom_dataset(custom_path)
        
        if self.dataset_name == "squad":
            return self.load_squad_dataset()
        elif self.dataset_name == "hotpot_qa":
            return self.load_hotpot_qa_dataset()
        elif self.dataset_name == "natural_questions":
            return self.load_natural_questions_dataset()
        else:
            logger.warning(f"Unsupported dataset: {self.dataset_name}, using sample data")
            return self._get_sample_documents()
    
    def save_processed_data(self, documents: List[Dict[str, str]], output_path: str):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(documents)} documents to {output_path}")
