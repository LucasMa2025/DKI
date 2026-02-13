"""
Experiment Data Generator for DKI System
Generates synthetic test data for experiments
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger


class ExperimentDataGenerator:
    """
    Generate synthetic experiment data for DKI testing.
    
    Datasets:
    - PersonaChat: Multi-turn dialogue with persona memories
    - HotpotQA: Multi-hop reasoning questions
    - MemoryQA: Memory recall benchmark
    """
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_persona_chat(
        self,
        n_sessions: int = 100,
        n_turns_per_session: int = 5,
        n_personas_per_session: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Generate PersonaChat-style dialogue data.
        
        Each session has:
        - Persona memories (user preferences, traits)
        - Multi-turn dialogue
        - Expected memory usage
        """
        personas = [
            "I love hiking and outdoor activities.",
            "I work as a software engineer at a tech company.",
            "I prefer vegetarian food and am allergic to seafood.",
            "I live in Beijing with my family.",
            "I enjoy photography as a hobby.",
            "I have a golden retriever named Max.",
            "I'm learning to play the guitar.",
            "I usually wake up early around 6 AM.",
            "I prefer coffee over tea.",
            "I'm planning a trip to Japan next year.",
            "I enjoy reading science fiction novels.",
            "I practice yoga every morning.",
            "I'm interested in artificial intelligence.",
            "I collect vintage watches.",
            "I'm a fan of classical music.",
        ]
        
        queries = [
            ("Can you recommend a restaurant for me?", ["food", "vegetarian", "seafood"]),
            ("What activities can I do this weekend?", ["hiking", "outdoor", "photography"]),
            ("Help me plan my morning routine.", ["wake up", "yoga", "coffee"]),
            ("What should I read next?", ["science fiction", "reading"]),
            ("Suggest a gift for my pet.", ["dog", "golden retriever", "Max"]),
            ("What music should I listen to while working?", ["classical music", "guitar"]),
            ("Where should I travel for vacation?", ["Japan", "trip", "travel"]),
            ("What skills should I learn?", ["guitar", "AI", "artificial intelligence"]),
            ("Recommend a workout routine.", ["yoga", "morning", "outdoor"]),
            ("What hobby should I pick up?", ["photography", "watches", "collecting"]),
        ]
        
        data = []
        
        for session_idx in range(n_sessions):
            session_id = f"persona_session_{session_idx:04d}"
            
            # Select random personas for this session
            session_personas = random.sample(personas, min(n_personas_per_session, len(personas)))
            
            # Generate turns
            turns = []
            used_queries = random.sample(queries, min(n_turns_per_session, len(queries)))
            
            for turn_idx, (query, expected_keywords) in enumerate(used_queries):
                # Find relevant personas
                relevant_memories = [
                    p for p in session_personas
                    if any(kw.lower() in p.lower() for kw in expected_keywords)
                ]
                
                turns.append({
                    'turn_id': turn_idx,
                    'query': query,
                    'expected_keywords': expected_keywords,
                    'relevant_memories': relevant_memories,
                })
            
            data.append({
                'session_id': session_id,
                'personas': session_personas,
                'turns': turns,
                'metadata': {
                    'dataset': 'persona_chat',
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        # Save to file
        output_path = self.output_dir / 'persona_chat.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} PersonaChat sessions to {output_path}")
        return data
    
    def generate_hotpot_qa(
        self,
        n_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate HotpotQA-style multi-hop reasoning data.
        
        Each sample has:
        - Multiple supporting facts
        - A question requiring reasoning across facts
        - Expected answer
        """
        fact_templates = [
            ("{person} was born in {city} in {year}.", ["person", "city", "year"]),
            ("{city} is the capital of {country}.", ["city", "country"]),
            ("{person} is famous for {achievement}.", ["person", "achievement"]),
            ("{company} was founded by {person} in {year}.", ["company", "person", "year"]),
            ("{city} is located in {country}.", ["city", "country"]),
            ("{person} works at {company} as a {role}.", ["person", "company", "role"]),
        ]
        
        entities = {
            "person": ["Alice Wang", "Bob Zhang", "Charlie Li", "Diana Chen", "Edward Liu"],
            "city": ["Beijing", "Shanghai", "Tokyo", "New York", "London"],
            "country": ["China", "Japan", "USA", "UK", "France"],
            "year": ["1990", "1995", "2000", "2005", "2010"],
            "company": ["TechCorp", "DataInc", "AILabs", "CloudSoft", "SmartSys"],
            "role": ["CEO", "Engineer", "Researcher", "Manager", "Designer"],
            "achievement": ["AI research", "founding a startup", "winning an award", "publishing papers"],
        }
        
        question_templates = [
            ("In which country was {person} born?", ["person", "city", "country"]),
            ("When was the company founded by {person}?", ["person", "company", "year"]),
            ("What is {person} famous for?", ["person", "achievement"]),
            ("Where does {person} work?", ["person", "company"]),
        ]
        
        data = []
        
        for i in range(n_samples):
            sample_id = f"hotpot_{i:04d}"
            
            # Generate random facts
            facts = []
            entity_values = {}
            
            for template, required_entities in random.sample(fact_templates, 2):
                for entity_type in required_entities:
                    if entity_type not in entity_values:
                        entity_values[entity_type] = random.choice(entities[entity_type])
                
                fact = template.format(**{k: entity_values.get(k, '') for k in required_entities})
                facts.append(fact)
            
            # Generate question
            # Filter templates to only those whose required entities are all available
            compatible_templates = [
                (qt, qe) for qt, qe in question_templates
                if all(e in entity_values for e in set(qt_var 
                       for qt_var in qe if '{' + qt_var + '}' in qt))
            ]
            if not compatible_templates:
                # Fallback: use first template and fill missing entities with defaults
                q_template, q_entities = question_templates[0]
            else:
                q_template, q_entities = random.choice(compatible_templates)
            
            # Provide all keys the template needs, with empty string fallback
            question = q_template.format(**{k: entity_values.get(k, 'Unknown') for k in q_entities})
            
            # Determine expected answer (simplified)
            expected_entities = [entity_values.get(e) for e in q_entities if e in entity_values]
            expected_answer = expected_entities[-1] if expected_entities else "Unknown"
            
            data.append({
                'id': sample_id,
                'question': question,
                'supporting_facts': facts,
                'expected_answer': expected_answer,
                'entity_values': entity_values,
                'metadata': {
                    'dataset': 'hotpot_qa',
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        # Save to file
        output_path = self.output_dir / 'hotpot_qa.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} HotpotQA samples to {output_path}")
        return data
    
    def generate_memory_qa(
        self,
        n_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate Memory QA benchmark data.
        
        Tests memory recall without explicit hints.
        """
        memory_templates = [
            {
                'memory': "My favorite color is {color}.",
                'query': "What color should I paint my room?",
                'expected_use': True,
                'vars': {'color': ['blue', 'green', 'red', 'yellow', 'purple']},
            },
            {
                'memory': "I'm allergic to {allergen}.",
                'query': "Can you suggest what to eat?",
                'expected_use': True,
                'vars': {'allergen': ['peanuts', 'shellfish', 'dairy', 'gluten']},
            },
            {
                'memory': "I live in {city}.",
                'query': "What's the weather like today?",
                'expected_use': True,
                'vars': {'city': ['Beijing', 'Shanghai', 'Tokyo', 'London', 'Paris']},
            },
            {
                'memory': "My birthday is on {date}.",
                'query': "What special events are coming up?",
                'expected_use': True,
                'vars': {'date': ['March 15', 'June 20', 'October 1', 'December 25']},
            },
            {
                'memory': "I prefer {preference} style.",
                'query': "Help me pick an outfit.",
                'expected_use': True,
                'vars': {'preference': ['casual', 'formal', 'sporty', 'minimalist']},
            },
        ]
        
        data = []
        
        for i in range(n_samples):
            template = random.choice(memory_templates)
            
            # Fill in variables
            filled_vars = {}
            for var_name, options in template['vars'].items():
                filled_vars[var_name] = random.choice(options)
            
            memory = template['memory'].format(**filled_vars)
            
            data.append({
                'id': f"memqa_{i:04d}",
                'memory': memory,
                'query': template['query'],
                'expected_memory_use': template['expected_use'],
                'filled_vars': filled_vars,
                'metadata': {
                    'dataset': 'memory_qa',
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        # Save to file
        output_path = self.output_dir / 'memory_qa.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} MemoryQA samples to {output_path}")
        return data
    
    def generate_all(
        self,
        persona_sessions: int = 100,
        hotpot_samples: int = 100,
        memory_qa_samples: int = 100,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all datasets."""
        return {
            'persona_chat': self.generate_persona_chat(persona_sessions),
            'hotpot_qa': self.generate_hotpot_qa(hotpot_samples),
            'memory_qa': self.generate_memory_qa(memory_qa_samples),
        }
    
    def generate_alpha_sensitivity_data(
        self,
        n_samples: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Generate data for α sensitivity analysis.
        
        Each sample tests different α values.
        """
        base_memories = [
            "User prefers vegetarian food and is allergic to seafood.",
            "User enjoys outdoor activities like hiking and photography.",
            "User works as a software engineer in Beijing.",
        ]
        
        queries = [
            "Recommend a restaurant for lunch.",
            "What should I do this weekend?",
            "Suggest a career development path.",
        ]
        
        alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        data = []
        
        for i in range(n_samples):
            memory = random.choice(base_memories)
            query = random.choice(queries)
            
            for alpha in alpha_values:
                data.append({
                    'id': f"alpha_{i:04d}_{int(alpha*100):02d}",
                    'memory': memory,
                    'query': query,
                    'alpha': alpha,
                    'metadata': {
                        'dataset': 'alpha_sensitivity',
                        'generated_at': datetime.now().isoformat(),
                    },
                })
        
        # Save to file
        output_path = self.output_dir / 'alpha_sensitivity.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} α sensitivity samples to {output_path}")
        return data


    def generate_chinese_persona_chat(
        self,
        n_sessions: int = 100,
        n_turns_per_session: int = 5,
        n_personas_per_session: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        生成中文 PersonaChat 对话数据
        
        每个会话包含:
        - 用户偏好记忆 (中文)
        - 多轮对话
        - 期望用到的记忆
        """
        personas = [
            "我喜欢徒步和户外运动。",
            "我是一名软件工程师，在科技公司工作。",
            "我是素食主义者，对海鲜过敏。",
            "我住在北京，和家人住在一起。",
            "我喜欢摄影，经常拍风景照。",
            "我养了一只金毛犬，叫小白。",
            "我正在学弹吉他。",
            "我通常早上六点起床。",
            "我喜欢喝咖啡，不太喝茶。",
            "我计划明年去日本旅行。",
            "我喜欢阅读科幻小说。",
            "我每天早上练瑜伽。",
            "我对人工智能很感兴趣。",
            "我喜欢收藏古董手表。",
            "我是古典音乐的爱好者。",
        ]
        
        queries = [
            ("能给我推荐一家餐厅吗？", ["素食", "海鲜", "过敏"]),
            ("这个周末我可以做什么活动？", ["徒步", "户外", "摄影"]),
            ("帮我规划一下早晨的日程。", ["起床", "瑜伽", "咖啡"]),
            ("我接下来应该读什么书？", ["科幻", "阅读"]),
            ("给我的宠物推荐个礼物。", ["金毛", "小白", "宠物"]),
            ("工作时应该听什么音乐？", ["古典音乐", "吉他"]),
            ("假期应该去哪里旅行？", ["日本", "旅行"]),
            ("我应该学习什么新技能？", ["吉他", "人工智能"]),
            ("推荐一个健身方案。", ["瑜伽", "户外", "运动"]),
            ("我应该培养什么新爱好？", ["摄影", "手表", "收藏"]),
        ]
        
        data = []
        
        for session_idx in range(n_sessions):
            session_id = f"cn_persona_session_{session_idx:04d}"
            session_personas = random.sample(personas, min(n_personas_per_session, len(personas)))
            
            turns = []
            used_queries = random.sample(queries, min(n_turns_per_session, len(queries)))
            
            for turn_idx, (query, expected_keywords) in enumerate(used_queries):
                relevant_memories = [
                    p for p in session_personas
                    if any(kw in p for kw in expected_keywords)
                ]
                
                turns.append({
                    'turn_id': turn_idx,
                    'query': query,
                    'expected_keywords': expected_keywords,
                    'relevant_memories': relevant_memories,
                })
            
            data.append({
                'session_id': session_id,
                'personas': session_personas,
                'turns': turns,
                'metadata': {
                    'dataset': 'cn_persona_chat',
                    'language': 'zh',
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        output_path = self.output_dir / 'cn_persona_chat.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} Chinese PersonaChat sessions to {output_path}")
        return data
    
    def generate_multi_turn_coherence(
        self,
        n_sessions: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        生成多轮连贯性测试数据
        
        测试 DKI 在多轮对话中的记忆保持和连贯性:
        - 早期轮次建立偏好信息
        - 中间轮次隐式引用
        - 后期轮次测试记忆保持
        """
        scenarios = [
            {
                'personas': [
                    "用户是素食主义者",
                    "用户住在上海浦东",
                    "用户有一只叫小花的猫",
                ],
                'turns': [
                    {'query': '你好，我想找个好吃的餐厅', 'tests_memory': False},
                    {'query': '我之前说过我的饮食习惯，帮我推荐吧', 'tests_memory': True, 'expected_recall': ['素食']},
                    {'query': '离我家近一点的', 'tests_memory': True, 'expected_recall': ['上海', '浦东']},
                    {'query': '对了，我想给我的宠物买个玩具', 'tests_memory': True, 'expected_recall': ['猫', '小花']},
                    {'query': '总结一下今天我们聊了什么', 'tests_memory': True, 'expected_recall': ['素食', '餐厅', '宠物']},
                ],
            },
            {
                'personas': [
                    "用户喜欢跑步和游泳",
                    "用户是程序员，熟悉Python",
                    "用户的生日是3月15日",
                ],
                'turns': [
                    {'query': '推荐一个周末活动', 'tests_memory': True, 'expected_recall': ['跑步', '游泳']},
                    {'query': '有什么技术书籍推荐吗', 'tests_memory': True, 'expected_recall': ['程序', 'Python']},
                    {'query': '下个月有什么特别的日子', 'tests_memory': True, 'expected_recall': ['生日', '3月']},
                    {'query': '帮我安排一个运动计划', 'tests_memory': True, 'expected_recall': ['跑步', '游泳']},
                    {'query': '你还记得我的职业吗', 'tests_memory': True, 'expected_recall': ['程序']},
                ],
            },
            {
                'personas': [
                    "用户对辣椒过敏",
                    "用户在北京工作",
                    "用户喜欢古典音乐",
                ],
                'turns': [
                    {'query': '午饭吃什么好', 'tests_memory': True, 'expected_recall': ['辣椒', '过敏']},
                    {'query': '推荐一个音乐会', 'tests_memory': True, 'expected_recall': ['古典音乐']},
                    {'query': '下班后去哪里吃好', 'tests_memory': True, 'expected_recall': ['北京', '辣椒']},
                    {'query': '你记得我不能吃什么吗', 'tests_memory': True, 'expected_recall': ['辣椒', '过敏']},
                    {'query': '帮我规划今晚的安排', 'tests_memory': True, 'expected_recall': ['音乐', '北京']},
                ],
            },
        ]
        
        data = []
        
        for i in range(n_sessions):
            scenario = scenarios[i % len(scenarios)]
            session_id = f"coherence_session_{i:04d}"
            
            data.append({
                'session_id': session_id,
                'personas': scenario['personas'],
                'turns': scenario['turns'],
                'metadata': {
                    'dataset': 'multi_turn_coherence',
                    'language': 'zh',
                    'scenario_idx': i % len(scenarios),
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        output_path = self.output_dir / 'multi_turn_coherence.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} multi-turn coherence sessions to {output_path}")
        return data
    
    def generate_ablation_data(
        self,
        n_samples: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        生成消融实验数据
        
        测试 DKI 各组件的贡献:
        - Full DKI (偏好 K/V + 历史后缀 + 门控)
        - No Gating (固定 α=1.0)
        - No History (仅偏好 K/V, 无历史后缀)
        - No Preference (仅历史后缀, 无偏好 K/V) 
        - RAG Baseline
        - No Memory Baseline
        """
        memories_cn = [
            "用户是素食主义者，不吃肉类和海鲜。",
            "用户住在北京海淀区，经常在中关村附近活动。",
            "用户喜欢户外运动，特别是徒步和骑行。",
            "用户是一名数据科学家，擅长机器学习。",
            "用户养了两只猫，叫花花和豆豆。",
        ]
        
        queries_cn = [
            {"query": "推荐一个适合我的午餐", "relevant_memory_idx": [0]},
            {"query": "周末去哪里玩比较好", "relevant_memory_idx": [1, 2]},
            {"query": "有什么新技术值得学习", "relevant_memory_idx": [3]},
            {"query": "给我的宠物买点什么", "relevant_memory_idx": [4]},
            {"query": "附近有什么好的运动场所", "relevant_memory_idx": [1, 2]},
        ]
        
        ablation_modes = [
            "full_dki",           # 完整 DKI
            "no_gating",          # 无门控 (固定 α=1.0)
            "no_history",         # 无历史后缀
            "no_preference_kv",   # 无偏好 K/V
            "rag_baseline",       # RAG 对照
            "no_memory",          # 无记忆基线
        ]
        
        data = []
        
        for i in range(n_samples):
            memory = random.choice(memories_cn)
            query_item = random.choice(queries_cn)
            
            relevant_memories = [memories_cn[idx] for idx in query_item["relevant_memory_idx"]]
            
            data.append({
                'id': f"ablation_{i:04d}",
                'memory': memory,
                'all_memories': memories_cn,
                'query': query_item['query'],
                'relevant_memories': relevant_memories,
                'ablation_modes': ablation_modes,
                'metadata': {
                    'dataset': 'ablation',
                    'language': 'zh',
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        output_path = self.output_dir / 'ablation.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} ablation experiment samples to {output_path}")
        return data
    
    def generate_all(
        self,
        persona_sessions: int = 100,
        hotpot_samples: int = 100,
        memory_qa_samples: int = 100,
        include_chinese: bool = True,
        include_advanced: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate all datasets."""
        result = {
            'persona_chat': self.generate_persona_chat(persona_sessions),
            'hotpot_qa': self.generate_hotpot_qa(hotpot_samples),
            'memory_qa': self.generate_memory_qa(memory_qa_samples),
        }
        
        if include_chinese:
            result['cn_persona_chat'] = self.generate_chinese_persona_chat(persona_sessions)
        
        if include_advanced:
            result['multi_turn_coherence'] = self.generate_multi_turn_coherence()
            result['ablation'] = self.generate_ablation_data()
        
        return result


def main():
    """Generate all experiment data."""
    generator = ExperimentDataGenerator("./data")
    generator.generate_all(include_chinese=True, include_advanced=True)
    generator.generate_alpha_sensitivity_data()
    print("Experiment data generated successfully!")


if __name__ == "__main__":
    main()
