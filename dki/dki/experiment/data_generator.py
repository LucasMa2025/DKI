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
        - experiment_user: 对应的实验用户名 (用于偏好注入)
        """
        # 按实验用户分组的 personas
        user_persona_groups = {
            "exp_user_vegetarian": [
                "I prefer vegetarian food and am allergic to seafood.",
                "I live in Beijing with my family.",
                "I prefer coffee over tea.",
                "I usually wake up early around 6 AM.",
                "I practice yoga every morning.",
            ],
            "exp_user_outdoor": [
                "I love hiking and outdoor activities.",
                "I enjoy photography as a hobby.",
                "I have a golden retriever named Max.",
                "I'm planning a trip to Japan next year.",
                "I live in Beijing with my family.",
            ],
            "exp_user_tech": [
                "I work as a software engineer at a tech company.",
                "I'm interested in artificial intelligence.",
                "I enjoy reading science fiction novels.",
                "I prefer coffee over tea.",
                "I usually wake up early around 6 AM.",
            ],
            "exp_user_music": [
                "I'm a fan of classical music.",
                "I'm learning to play the guitar.",
                "I collect vintage watches.",
                "I enjoy photography as a hobby.",
                "I live in Beijing with my family.",
            ],
        }
        
        # 所有 personas (保持向后兼容)
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
        user_names = list(user_persona_groups.keys())
        
        for session_idx in range(n_sessions):
            session_id = f"persona_session_{session_idx:04d}"
            
            # 轮流分配实验用户，确保每个用户有足够的样本
            assigned_user = user_names[session_idx % len(user_names)]
            
            # 从该用户的 persona 组中选择 (优先)，不足时从全局补充
            user_personas = user_persona_groups[assigned_user]
            if len(user_personas) >= n_personas_per_session:
                session_personas = random.sample(user_personas, n_personas_per_session)
            else:
                session_personas = list(user_personas)
                remaining = n_personas_per_session - len(session_personas)
                extra = [p for p in personas if p not in session_personas]
                session_personas.extend(random.sample(extra, min(remaining, len(extra))))
            
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
                'experiment_user': assigned_user,  # 对应的实验用户名
                'personas': session_personas,
                'turns': turns,
                'metadata': {
                    'dataset': 'persona_chat',
                    'experiment_user': assigned_user,
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
        - experiment_user: 对应的实验用户名
        """
        # 按实验用户分组的中文 personas
        cn_user_persona_groups = {
            "exp_user_vegetarian": [
                "我是素食主义者，对海鲜过敏。",
                "我住在北京，和家人住在一起。",
                "我喜欢喝咖啡，不太喝茶。",
                "我通常早上六点起床。",
                "我每天早上练瑜伽。",
            ],
            "exp_user_outdoor": [
                "我喜欢徒步和户外运动。",
                "我喜欢摄影，经常拍风景照。",
                "我养了一只金毛犬，叫小白。",
                "我计划明年去日本旅行。",
                "我住在北京，和家人住在一起。",
            ],
            "exp_user_tech": [
                "我是一名软件工程师，在科技公司工作。",
                "我对人工智能很感兴趣。",
                "我喜欢阅读科幻小说。",
                "我喜欢喝咖啡，不太喝茶。",
                "我通常早上六点起床。",
            ],
            "exp_user_music": [
                "我是古典音乐的爱好者。",
                "我正在学弹吉他。",
                "我喜欢收藏古董手表。",
                "我喜欢摄影，经常拍风景照。",
                "我住在北京，和家人住在一起。",
            ],
        }
        
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
        cn_user_names = list(cn_user_persona_groups.keys())
        
        for session_idx in range(n_sessions):
            session_id = f"cn_persona_session_{session_idx:04d}"
            
            # 轮流分配实验用户
            assigned_user = cn_user_names[session_idx % len(cn_user_names)]
            
            # 从该用户的 persona 组中选择
            user_personas = cn_user_persona_groups[assigned_user]
            if len(user_personas) >= n_personas_per_session:
                session_personas = random.sample(user_personas, n_personas_per_session)
            else:
                session_personas = list(user_personas)
                remaining = n_personas_per_session - len(session_personas)
                extra = [p for p in personas if p not in session_personas]
                session_personas.extend(random.sample(extra, min(remaining, len(extra))))
            
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
                'experiment_user': assigned_user,
                'personas': session_personas,
                'turns': turns,
                'metadata': {
                    'dataset': 'cn_persona_chat',
                    'language': 'zh',
                    'experiment_user': assigned_user,
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
    
    def generate_long_session_persona_chat(
        self,
        n_sessions: int = 20,
        n_turns_per_session: int = 15,
        min_turn_length: int = 512,
        max_turn_length: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        生成长会话 PersonaChat 数据 — 强调 DKI 优势场景
        
        每个会话:
        - 15+ 轮对话
        - 每轮输入/输出 512-2K 字符长度
        - 包含详细的上下文、背景描述和复杂查询
        - 测试 DKI 在长上下文、高记忆压力下的优势
        
        设计理念:
        - RAG 在长上下文下会挤压推理 token 预算
        - DKI 通过 K/V 注入绕过偏好记忆的 token 消耗
        - 长会话累积的历史使 multi-signal recall 优势更明显
        """
        # 长会话场景模板 — 每个场景有丰富的上下文和多步推理需求
        long_session_scenarios = [
            {
                "experiment_user": "exp_user_vegetarian",
                "personas": [
                    "我是严格的素食主义者，已经坚持素食15年了。我对所有动物制品都非常敏感，包括蛋奶制品。我住在北京海淀区中关村附近，经常在五道口和学院路一带活动。",
                    "我有严重的海鲜过敏史，曾经因为误食含有虾仁的菜品而住院过。医生建议我随身携带抗过敏药物。任何含有海鲜成分的食物都可能引发我的过敏反应。",
                    "我是一名营养学研究者，在中国农业大学工作。我对食品安全和营养搭配有专业的了解，经常需要为自己和学生设计营养均衡的素食食谱。",
                    "我每周末会去有机农场采购新鲜蔬果，对食材的来源和种植方式非常讲究。我偏好本地种植的时令蔬菜，反对过度加工的食品。",
                    "我正在写一本关于中国传统素食文化的书，需要收集各地素食餐厅的信息和特色菜品。这本书计划明年在清华大学出版社出版。",
                ],
                "turns": [
                    {
                        "query": "你好，我最近在研究北京地区的素食餐厅分布情况，想请你帮我做一个详细的分析。我需要了解海淀区、朝阳区和东城区三个区域的素食餐厅数量、类型（纯素、蛋奶素、佛教素食等）以及价位分布。如果可能的话，还想了解这些餐厅的食材来源是否可追溯，是否有有机认证。这个分析将用于我正在写的关于中国素食文化的书籍的第三章。",
                        "expected_keywords": ["素食", "海淀", "餐厅"],
                        "expected_length_range": [512, 2048],
                    },
                    {
                        "query": "谢谢你的分析。基于我之前提到的过敏情况，能否帮我进一步筛选出那些能够保证完全不含海鲜成分的餐厅？我之前有过一次非常严重的过敏经历，是在一家号称纯素的餐厅吃到了用蚝油调味的菜品。所以我特别需要那些能够明确标注所有调味料成分的餐厅。另外，我想了解这些餐厅是否提供定制化的菜单服务，因为我有时候需要为我的研究团队组织聚餐。",
                        "expected_keywords": ["过敏", "海鲜", "素食"],
                        "expected_length_range": [512, 2048],
                    },
                    {
                        "query": "说到营养搭配，我最近在为我的新学期课程准备一个关于'植物性蛋白质完全互补'的教学案例。我需要设计一周的素食食谱，要求每天的蛋白质摄入量不低于60克，同时要考虑到必需氨基酸的完整性。食谱需要使用北京当季可获得的食材，价格要合理（每人每天预算不超过80元），而且要兼顾口味的多样性。你能帮我设计这个食谱吗？请详细列出每餐的食材、用量和烹饪方法。",
                        "expected_keywords": ["素食", "营养", "蛋白质"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "非常好的建议。现在我想讨论一个更深入的话题——关于我书中第五章要写的'中国传统佛教素食与现代素食主义的对话'。我需要从历史、哲学和营养学三个角度来分析这个话题。历史方面，我想追溯从南北朝时期梁武帝推行素食到现代的发展脉络；哲学方面，需要比较佛教慈悲理念与现代动物权利运动的异同；营养学方面，要分析传统佛教素食（如豆腐、面筋等）的营养价值与现代植物肉的对比。请帮我梳理这个章节的大纲和关键论点。",
                        "expected_keywords": ["素食", "佛教", "文化"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "回到实际生活，这个周末我计划去昌平的有机农场采购食材。我通常会采购一周的量，大概需要以下几类：深色叶菜（菠菜、油菜、芥蓝）、根茎类（胡萝卜、莲藕、山药）、豆类（毛豆、黄豆、黑豆）、菌菇类（香菇、杏鲍菇、金针菇）。考虑到现在是冬季，有些蔬菜可能不是当季的。你能帮我制定一个采购清单吗？要考虑到储存条件（我家冰箱容量有限）和保鲜期限，以及这些食材可以搭配做出哪些菜品。",
                        "expected_keywords": ["有机", "农场", "采购"],
                        "expected_length_range": [512, 2048],
                    },
                    {
                        "query": "对了，我下周三要在学校组织一个素食文化讲座，预计有50人参加。讲座结束后需要提供一个素食自助餐。预算是每人50元，需要你帮我设计菜单。要求：1）完全纯素，不含任何动物制品和蚝油等动物性调味料；2）至少8道菜品，包含冷菜、热菜、汤品和甜点；3）要考虑到可能有人对坚果或大豆过敏；4）菜品要有中国特色，因为有几位外国留学生参加；5）需要提前联系供应商，你能推荐海淀区可靠的素食catering服务吗？",
                        "expected_keywords": ["素食", "讲座", "海淀"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "讲座的内容我也想和你讨论一下。主题是'从餐桌到实验室：素食营养学的前沿研究'。我计划分三个部分：第一部分介绍素食营养学的基本原理和常见误区（比如'素食者一定缺铁'这种说法）；第二部分分享我最近在维生素B12和omega-3脂肪酸补充方面的研究发现；第三部分是互动环节，准备了一个'素食营养知识问答'。你能帮我准备第一部分的演讲大纲吗？需要有数据支撑，引用权威的营养学研究。",
                        "expected_keywords": ["营养", "素食", "研究"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "说到研究，我最近收到了一个来自《中国营养学报》的审稿邀请，是一篇关于植物性饮食对肠道菌群影响的论文。作为审稿人，我需要评估这篇论文的实验设计是否合理。论文声称'纯素饮食6个月后，受试者肠道中双歧杆菌和乳酸菌的丰度显著增加'。你能帮我列出在审稿时应该关注的关键方法学问题吗？比如样本量、对照组设计、混杂因素控制、统计方法等。",
                        "expected_keywords": ["研究", "营养", "素食"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "转换一下话题。我的书稿编辑建议我增加一个关于'全球素食产业发展趋势'的附录。我需要收集以下数据：1）全球植物肉市场规模和增长率（2020-2025）；2）中国素食人口的估计数量和增长趋势；3）主要的植物肉品牌在中国的市场表现；4）消费者对植物肉的接受度调查数据。你能帮我整理这些信息吗？我知道你可能没有最新的数据，但请提供你所知道的信息和可靠的数据来源建议。",
                        "expected_keywords": ["素食", "产业", "趋势"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "最后，我想请你帮我回顾一下我们今天讨论的所有内容，按照以下结构整理：1）我的个人饮食需求和限制（包括过敏信息）；2）北京素食餐厅分析的要点；3）一周食谱设计的核心原则；4）书稿相关的研究方向和待完成任务；5）下周讲座的准备事项。请确保不要遗漏任何重要信息，特别是与我的过敏情况相关的安全提醒。",
                        "expected_keywords": ["素食", "过敏", "总结"],
                        "expected_length_range": [1024, 2048],
                    },
                ],
            },
            {
                "experiment_user": "exp_user_tech",
                "personas": [
                    "我是一名资深数据科学家，在一家AI创业公司担任技术负责人。我擅长Python、PyTorch和TensorFlow，有8年的机器学习实战经验。最近在做大语言模型的微调和部署工作。",
                    "我对分布式系统和高性能计算有深入研究，特别是在GPU集群调度和模型并行训练方面。我们团队使用的是8卡A100集群。",
                    "我喜欢阅读科幻小说，尤其是刘慈欣和阿西莫夫的作品。我认为科幻文学对AI研究者的想象力培养非常重要。",
                    "我住在北京望京，每天通勤大约40分钟。我喜欢在通勤路上听技术播客，最近在听Lex Fridman的节目。",
                    "我正在准备一个关于'LLM推理优化'的技术分享，计划在下个月的公司技术日上演讲。",
                ],
                "turns": [
                    {
                        "query": "你好，我最近在研究如何优化我们公司的LLM推理服务。目前我们使用vLLM部署了一个7B参数的模型，但在高并发场景下（QPS > 100）延迟显著上升。我想从以下几个方面进行优化：1）KV Cache管理策略；2）批处理调度算法；3）量化方案选择（INT8 vs FP8 vs GPTQ）；4）Tensor并行 vs Pipeline并行的权衡。请帮我分析每个方面的优缺点和适用场景，最好能给出一些具体的配置建议。我们的硬件是8卡A100 80G。",
                        "expected_keywords": ["LLM", "优化", "推理"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "谢谢分析。关于KV Cache管理，我注意到vLLM的PagedAttention虽然解决了内存碎片问题，但在我们的场景下仍然存在一些问题。具体来说，我们有很多长上下文的请求（4K-8K tokens），这些请求的KV Cache占用了大量GPU内存，导致短请求的排队时间增加。我在考虑实现一个优先级调度器，根据请求的预估token数和等待时间来动态分配GPU资源。你能帮我设计这个调度算法吗？需要考虑公平性、延迟SLA和GPU利用率的平衡。",
                        "expected_keywords": ["KV Cache", "调度", "GPU"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "说到量化，我最近在测试AWQ和GPTQ两种量化方案。在我们的基准测试中，AWQ-INT4在大多数任务上的精度损失小于1%，但在数学推理任务上损失了约3%。GPTQ-INT4的整体精度略低，但推理速度快了约15%。考虑到我们的业务场景中数学推理占比约20%，你建议我选择哪种方案？或者，是否应该考虑混合精度策略——对attention层使用FP16，对FFN层使用INT4？请从工程实现的角度分析可行性。",
                        "expected_keywords": ["量化", "精度", "推理"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "现在转到模型微调的话题。我们团队正在用LoRA对一个13B的模型进行领域微调，训练数据是我们自己标注的10万条客服对话。目前的问题是：1）训练loss在第3个epoch后开始震荡，不再下降；2）模型在训练集上表现很好，但在测试集上的BLEU分数只有0.35；3）生成的回复有时会出现重复和不连贯的情况。我怀疑是过拟合了，但也不确定是不是LoRA的rank设置（目前r=16）不合适。你能帮我诊断这些问题并给出调优建议吗？",
                        "expected_keywords": ["微调", "LoRA", "训练"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "对了，我下个月的技术分享主题确定为'从理论到实践：LLM推理优化的工程之路'。我计划用45分钟的时间，覆盖以下内容：1）LLM推理的计算瓶颈分析（memory-bound vs compute-bound）；2）主流优化技术综述（FlashAttention、PagedAttention、Speculative Decoding等）；3）我们团队的实践经验和踩过的坑；4）未来展望（MoE、稀疏注意力等）。你能帮我设计一个详细的演讲大纲吗？每个部分大约需要多少时间？哪些地方适合加入demo？",
                        "expected_keywords": ["技术分享", "LLM", "优化"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "周末在听Lex Fridman采访Andrej Karpathy的那期播客时，他提到了一个很有意思的观点：'The best way to understand neural networks is to build them from scratch.' 这让我想到，也许我应该在技术分享中加入一个live coding环节，从零实现一个简化版的KV Cache管理器。你觉得这个想法怎么样？如果要实现，核心代码大概需要多少行？应该包含哪些关键功能（分配、释放、LRU淘汰、碎片整理）？",
                        "expected_keywords": ["播客", "coding", "KV Cache"],
                        "expected_length_range": [512, 2048],
                    },
                    {
                        "query": "最近读完了刘慈欣的《球状闪电》，里面关于量子态宏观物体的设想让我联想到了一个有趣的类比：LLM中的attention机制是否可以类比为'观测坍缩'——在所有可能的token中，attention权重的分配就像是一种'观测'，将概率分布坍缩为具体的输出。这个类比当然不严格，但我想在技术分享的开场用这个故事来引入attention的概念。你能帮我把这个类比展开得更生动一些吗？同时请指出这个类比的局限性，避免误导听众。",
                        "expected_keywords": ["科幻", "attention", "类比"],
                        "expected_length_range": [512, 2048],
                    },
                    {
                        "query": "回到工程问题。我们的A100集群最近遇到了一个棘手的问题：在使用Tensor Parallel=4进行推理时，NCCL通信偶尔会出现timeout，导致整个推理请求失败。这个问题在高负载时更频繁（大约每1000个请求出现1次）。我已经检查了网络配置（使用的是InfiniBand），NCCL版本是2.18.1，CUDA是12.1。你能帮我排查这个问题吗？需要检查哪些日志和配置？是否有已知的NCCL bug与此相关？",
                        "expected_keywords": ["GPU", "集群", "NCCL"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "综合我们今天讨论的所有技术问题，请帮我整理一份'LLM推理优化路线图'。按照优先级排序，列出每个优化项的预期收益、实施难度、所需时间和风险。考虑到我们团队只有3个人（包括我），资源有限，需要做好取舍。另外，请基于我之前提到的硬件配置（8卡A100）和业务场景（高并发客服），给出一个3个月的里程碑计划。",
                        "expected_keywords": ["优化", "路线图", "计划"],
                        "expected_length_range": [1024, 2048],
                    },
                ],
            },
            {
                "experiment_user": "exp_user_outdoor",
                "personas": [
                    "我是一名资深户外运动爱好者，有10年的徒步和登山经验。我完成过四姑娘山大峰、哈巴雪山和慕士塔格峰的攀登。我住在上海浦东，但经常飞往各地进行户外活动。",
                    "我养了一只3岁的金毛犬叫小白，它是我的户外伙伴。我经常带它去崇明岛和佘山徒步。小白有轻微的髋关节发育不良，需要注意运动强度。",
                    "我是一名自由摄影师，专注于户外风光和野生动物摄影。我使用的是索尼A7R4和一套从16mm到600mm的镜头组合。最近在拍摄一个关于长江入海口湿地鸟类的专题。",
                    "我计划明年春天去日本进行一次为期两周的徒步摄影之旅，主要目标是拍摄樱花季的富士山和北海道的丹顶鹤。",
                    "我对环保和可持续发展非常关注，是Leave No Trace原则的践行者。我在小红书上运营一个有5万粉丝的户外环保账号。",
                ],
                "turns": [
                    {
                        "query": "你好，我正在规划明年春天的日本徒步摄影之旅，需要你帮我做一个详细的行程规划。时间是3月底到4月中旬，共14天。我的目标是：1）拍摄富士山和樱花的经典构图（河口湖、忍野八海、新�的五合目）；2）在北海道拍摄丹顶鹤（�的路湿原）；3）沿途完成至少3条经典徒步路线。预算大约5万人民币（不含机票）。请帮我规划每天的行程，包括住宿、交通、拍摄时间窗口和备选方案。注意，我会带我的金毛犬小白一起去，所以需要考虑宠物友好的住宿和交通。",
                        "expected_keywords": ["日本", "徒步", "摄影", "小白"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "关于带小白去日本的问题，我需要了解更多细节。小白有髋关节发育不良的情况，兽医建议每天的运动量不超过10公里，而且要避免陡峭的上下坡。在日本的徒步路线中，哪些是比较平缓适合带狗的？另外，日本对宠物入境有哪些检疫要求？我需要提前多久准备？小白目前的疫苗接种记录是完整的（狂犬、六联），最近一次体检是上个月，一切正常。还有，日本的公共交通（新干线、巴士）允许带宠物吗？有什么注意事项？",
                        "expected_keywords": ["小白", "日本", "宠物", "髋关节"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "现在讨论摄影装备的问题。对于这次日本之旅，我在考虑是否需要更换一些装备。我目前的镜头配置是：Sony 16-35mm f/2.8 GM、24-70mm f/2.8 GM、70-200mm f/2.8 GM和100-400mm f/4.5-5.6 GM，加上一个1.4x增距镜用于拍摄丹顶鹤。问题是：1）拍摄丹顶鹤时100-400mm+1.4x（等效560mm）够不够？是否需要租一个600mm f/4？2）拍摄樱花时需要微距镜头吗？3）三脚架我有一个碳纤维的Gitzo GT3543LS，但它太重了（2.8kg），是否应该换一个更轻的旅行三脚架？请从实用性和重量的角度给我建议。",
                        "expected_keywords": ["摄影", "镜头", "装备"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "说到我的湿地鸟类摄影项目，我最近在崇明东滩拍到了一群黑脸琵鹭，这是一个意外的收获。我想把这次日本之旅也纳入到这个项目中——拍摄丹顶鹤可以作为'东亚-澳大利西亚候鸟迁飞路线'专题的一部分。你能帮我梳理一下这个专题的框架吗？我想从以下角度来构建：1）候鸟迁飞路线的科学背景；2）关键栖息地的现状和保护挑战；3）我在各地拍摄的影像记录。这个项目最终的目标是在上海自然博物馆做一个摄影展。",
                        "expected_keywords": ["鸟类", "摄影", "湿地"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "关于环保方面，我想在这次日本之旅中实践'零废弃徒步'的理念，并在我的小红书账号上记录全过程。具体来说：1）如何在14天的旅途中尽量减少一次性用品的使用？2）日本的垃圾分类规则是怎样的？我需要随身携带哪些分类袋？3）在野外徒步时如何处理食物残渣和人类排泄物？4）有没有日本本地的Leave No Trace组织可以联系？我想在旅途中拜访他们。请帮我制定一个详细的'零废弃旅行计划'。",
                        "expected_keywords": ["环保", "零废弃", "Leave No Trace"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "我想为小红书准备一系列关于这次日本之旅的内容。计划发布15-20条帖子，覆盖行前准备、途中记录和旅后总结。请帮我设计一个内容日历，包括每条帖子的主题、配图建议和关键话题标签。考虑到我的账号定位是'户外环保'，内容需要平衡'美景分享'和'环保理念传播'。另外，我想在其中穿插一些关于小白的内容（它第一次出国的经历），因为宠物相关的内容通常互动率更高。",
                        "expected_keywords": ["小红书", "内容", "小白"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "回到徒步路线的规划。在日本期间，我想完成以下三条路线：1）富士山周边的'富士山一周步道'（部分段落）；2）北海道的'知床五湖步道'；3）京都的'东山步道'（哲学之道延伸）。请帮我评估每条路线的难度、所需时间、海拔变化和最佳拍摄点。特别注意，我需要带小白同行，所以要确认哪些路线允许带狗，以及是否有需要绕行的段落。还要考虑3-4月份的天气情况和可能遇到的野生动物（熊？）。",
                        "expected_keywords": ["徒步", "路线", "日本", "小白"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "请帮我总结一下我们讨论的所有内容，整理成一份完整的'日本徒步摄影之旅计划书'。包括：1）行程概览（14天日程表）；2）摄影装备清单和租赁计划；3）小白的出行准备（检疫、交通、住宿）；4）环保实践方案；5）小红书内容计划；6）预算明细；7）应急预案（天气变化、小白身体状况、装备故障等）。请确保所有之前讨论的细节都被包含在内，特别是小白的髋关节问题和我的摄影项目需求。",
                        "expected_keywords": ["总结", "计划", "小白", "摄影"],
                        "expected_length_range": [1024, 2048],
                    },
                ],
            },
            {
                "experiment_user": "exp_user_music",
                "personas": [
                    "我是一名古典音乐发烧友，尤其热爱贝多芬和莫扎特的作品。我收藏了超过500张黑胶唱片，其中最珍贵的是一张1962年Glenn Gould演奏的贝多芬钢琴奏鸣曲。",
                    "我正在学习古典吉他，目前在练习Giuliani的大序曲和Villa-Lobos的练习曲。我的老师是上海音乐学院的一位退休教授。",
                    "我对辣椒严重过敏，即使是微量的辣椒素也会引起严重的皮肤反应和呼吸困难。在外就餐时需要特别谨慎。",
                    "我在北京国贸附近的一家金融公司工作，工作压力很大。音乐是我最重要的减压方式。我每周至少去一次音乐会。",
                    "我最近在研究音乐治疗的科学原理，考虑在业余时间考取音乐治疗师的资格证书。",
                ],
                "turns": [
                    {
                        "query": "你好，我最近在深入研究贝多芬晚期钢琴奏鸣曲（Op.106-111）的演绎传统。我想比较几位伟大钢琴家对这些作品的不同诠释：Schnabel、Backhaus、Brendel、Pollini和Barenboim。请帮我分析每位钢琴家的演绎特点，特别是在以下方面：1）速度选择和节奏弹性；2）力度层次和音色控制；3）对贝多芬晚期风格的理解（形式创新、精神深度）；4）录音质量和历史背景。我计划写一篇比较文章发在我的音乐博客上。",
                        "expected_keywords": ["贝多芬", "钢琴", "演绎"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "说到我的吉他学习，我最近在练习Villa-Lobos的第一号练习曲（Arpeggio练习），但遇到了一些技术问题。具体来说：1）右手的p-i-m-a指法在快速琶音时不够均匀，尤其是a指（无名指）力度偏弱；2）左手在高把位的横按和弦（barre chord）时手指容易疲劳；3）整体音色不够温暖，感觉太'干'。我的老师建议我调整右手的触弦角度，但我不太理解具体应该怎么做。你能从技术角度帮我分析这些问题并给出练习建议吗？",
                        "expected_keywords": ["吉他", "练习", "Villa-Lobos"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "这个周末北京有几场音乐会我都想去，但时间冲突了。选项A是国家大剧院的马勒第五交响曲（余隆指挥中国爱乐），选项B是中山音乐堂的巴赫无伴奏大提琴组曲（王健独奏），选项C是保利剧院的爵士之夜（Wynton Marsalis来京巡演）。考虑到我的音乐偏好和最近的心情（工作压力大，需要放松），你推荐我去哪一场？请详细分析每场音乐会的亮点和可能的体验。另外，音乐会结束后我想在附近找一家安静的餐厅吃晚饭，记住我对辣椒过敏，不能吃任何辣的食物。",
                        "expected_keywords": ["音乐会", "辣椒过敏", "北京"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "关于音乐治疗的学习，我想了解更多关于认证路径的信息。我目前的背景是：金融从业者（非音乐专业），但有扎实的音乐理论基础（通过了英国皇家音乐学院的8级乐理考试）和多年的演奏经验。在中国，音乐治疗师的认证体系是怎样的？需要哪些前置课程？是否需要心理学或医学背景？有哪些推荐的培训机构？考虑到我是在职学习，有没有在线或周末的课程？预计需要多长时间才能获得资格？",
                        "expected_keywords": ["音乐治疗", "认证", "学习"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "我最近在整理我的黑胶唱片收藏，想建立一个数字化的目录系统。我有超过500张唱片，涵盖古典、爵士和少量摇滚。我想记录每张唱片的以下信息：1）基本信息（标题、艺术家、唱片公司、发行年份、编号）；2）品相评级（使用Goldmine标准）；3）购买信息（时间、地点、价格）；4）播放记录和个人评分；5）市场估值。你能帮我设计一个数据库结构吗？我打算用Python+SQLite来实现。另外，有没有现成的唱片管理软件可以推荐？",
                        "expected_keywords": ["黑胶唱片", "收藏", "数据库"],
                        "expected_length_range": [1024, 2048],
                    },
                    {
                        "query": "请帮我总结我们今天讨论的所有内容。特别注意：1）关于贝多芬晚期奏鸣曲的比较分析要点；2）吉他练习的技术建议；3）周末音乐会的推荐和餐厅选择（记住我的辣椒过敏）；4）音乐治疗认证的学习路径；5）唱片收藏管理系统的设计方案。请按照优先级排序，告诉我接下来应该先做哪些事情。",
                        "expected_keywords": ["总结", "辣椒过敏", "音乐"],
                        "expected_length_range": [1024, 2048],
                    },
                ],
            },
        ]
        
        data = []
        
        for session_idx in range(n_sessions):
            scenario = long_session_scenarios[session_idx % len(long_session_scenarios)]
            session_id = f"long_session_{session_idx:04d}"
            
            turns = []
            for turn_idx, turn_data in enumerate(scenario["turns"][:n_turns_per_session]):
                turns.append({
                    'turn_id': turn_idx,
                    'query': turn_data["query"],
                    'expected_keywords': turn_data["expected_keywords"],
                    'expected_length_range': turn_data.get("expected_length_range", [512, 2048]),
                    'relevant_memories': [
                        p for p in scenario["personas"]
                        if any(kw.lower() in p.lower() for kw in turn_data["expected_keywords"])
                    ],
                })
            
            data.append({
                'session_id': session_id,
                'session_type': 'long',  # 标记为长会话
                'experiment_user': scenario["experiment_user"],
                'personas': scenario["personas"],
                'turns': turns,
                'metadata': {
                    'dataset': 'long_session_persona_chat',
                    'session_type': 'long',
                    'language': 'zh',
                    'min_turn_length': min_turn_length,
                    'max_turn_length': max_turn_length,
                    'experiment_user': scenario["experiment_user"],
                    'generated_at': datetime.now().isoformat(),
                },
            })
        
        # Save to file
        output_path = self.output_dir / 'long_session_persona_chat.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated {len(data)} long-session PersonaChat sessions to {output_path}")
        return data

    def generate_all(
        self,
        persona_sessions: int = 100,
        hotpot_samples: int = 100,
        memory_qa_samples: int = 100,
        include_chinese: bool = True,
        include_advanced: bool = True,
        include_long_sessions: bool = True,
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
        
        if include_long_sessions:
            result['long_session_persona_chat'] = self.generate_long_session_persona_chat()
        
        return result


def main():
    """Generate all experiment data."""
    generator = ExperimentDataGenerator("./data")
    generator.generate_all(include_chinese=True, include_advanced=True, include_long_sessions=True)
    generator.generate_alpha_sensitivity_data()
    print("Experiment data generated successfully!")


if __name__ == "__main__":
    main()
