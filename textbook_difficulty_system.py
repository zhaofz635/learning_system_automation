import pandas as pd
import numpy as np
import json
import joblib
import os
import re
import math
from datetime import datetime
import requests
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
from skimage.feature import graycomatrix, graycoprops

# ========================================
# 🔧 NLTK 路径与资源管理（关键修复）
# ========================================

# 定义与 GitHub Actions 一致的路径
NLTK_DATA_PATH = '/tmp/nltk_data'

def setup_nltk():
    """确保 NLTK 能找到数据目录"""
    # 优先使用环境变量
    custom_path = os.getenv('NLTK_DATA')
    if custom_path and custom_path not in nltk.data.path:
        nltk.data.path.insert(0, custom_path)
    
    # 添加默认路径
    if NLTK_DATA_PATH not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_PATH)
    
    # 调试输出
    print("🔍 NLTK data search paths:", nltk.data.path)

def download_nltk_resources():
    """下载必要资源到指定目录"""
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ punkt 已存在")
    except LookupError:
        print(f"📥 正在下载 punkt 到 {NLTK_DATA_PATH}")
        nltk.download('punkt', download_dir=NLTK_DATA_PATH, quiet=False)

    try:
        nltk.data.find('corpora/stopwords')
        print("✅ stopwords 已存在")
    except LookupError:
        print(f"📥 正在下载 stopwords 到 {NLTK_DATA_PATH}")
        nltk.download('stopwords', download_dir=NLTK_DATA_PATH, quiet=False)

# 执行初始化
setup_nltk()
download_nltk_resources()

# ========================================
# 📚 术语库
# ========================================

class AcademicTermBank:
    def __init__(self, path="academic_terms.txt"):
        self.terms = self.load_terms(path)

    def load_terms(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                terms = set(f.read().splitlines())
            print(f"术语库加载成功，术语总数: {len(terms)}")
            return terms
        except FileNotFoundError:
            print(f"未找到术语库文件: {path}, 使用空术语库")
            return set()

# ========================================
# 🖼️ 表格/图像复杂度分析器
# ========================================

class TableComplexityAnalyzer:
    def __init__(self, weight_structured=0.6, weight_image=0.4):
        self.weights = {'structured': weight_structured, 'image': weight_image}
        self.sub_weights = {'row_complexity': 0.4, 'method_diversity': 0.3, 'column_variety': 0.3}
        self.vectorizer = TfidfVectorizer()
        self._prepare_tfidf()

    def _prepare_tfidf(self):
        IMPORTANT_COLUMNS = ['metric', 'formula', 'threshold', 'method']
        self.vectorizer.fit(IMPORTANT_COLUMNS)

    def analyze_entry(self, entry):
        struct_score = self._calc_structured_features(entry)
        image_score = self._calc_image_features(entry.get('image_path', ''))
        score = np.round(struct_score * self.weights['structured'] + image_score * self.weights['image'], 2)
        if score == 0 and 'text' in entry:
            text = entry['text']
            words = word_tokenize(text)
            if len(words) > 100:
                score = 0.5
        return score

    def _calc_structured_features(self, entry):
        if 'structured_data' not in entry or not entry['structured_data']:
            return 0
        table = entry['structured_data']
        rows = table.get('rows', [])
        row_comp = sum(len(r.get('calculation', {}).get('methods', [])) for r in rows) / len(rows) if rows else 0
        methods = {m['name'] for r in rows for m in r.get('calculation', {}).get('methods', [])}
        method_div = len(methods)
        columns = table.get('columns', [])
        if columns:
            X = self.vectorizer.transform(columns)
            col_var = X.mean(axis=1).sum() * 0.5
        else:
            col_var = 0
        return (row_comp * self.sub_weights['row_complexity'] +
                method_div * self.sub_weights['method_diversity'] +
                col_var * self.sub_weights['column_variety'])

    def _calc_image_features(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return 0.0
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            img = cv2.resize(img, (256, 256))
            img_eq = cv2.equalizeHist(img)
            img_bin = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            glcm = graycomatrix(img_bin, distances=[3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            return contrast * 0.7 + dissimilarity * 0.3
        except Exception as e:
            print(f"图像处理错误: {str(e)}")
            return 0.0

# ========================================
# 📊 计算各项复杂度
# ========================================

def calculate_language_complexity(text, term_bank):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    stop_words = set(stopwords.words("english"))
    non_stop_words = [w for w in words if w.lower() not in stop_words]
    non_stop_ratio = len(non_stop_words) / max(len(words), 1)
    long_words = [w for w in words if len(w) >= 6]
    long_word_ratio = len(long_words) / max(len(words), 1)
    tokens = [w.lower() for w in words]
    ngrams = set()
    for n in range(1, 4):
        for i in range(len(tokens) - n + 1):
            ngrams.add(' '.join(tokens[i:i + n]))
    matched_terms = ngrams & term_bank.terms
    term_ratio = min(1.0, len(matched_terms) / 8.0)
    sentence_length_factor = min(avg_sentence_length / 20, 1.0)
    w1, w2, w3, w4 = 0.3, 0.3, 0.3, 0.1
    complex_word_ratio = (
            w1 * non_stop_ratio +
            w2 * long_word_ratio +
            w3 * term_ratio +
            w4 * sentence_length_factor
    )
    return min(max(complex_word_ratio, 0.0), 1.0)

def calculate_formula_density(text):
    formula_pattern = re.compile(
        r"\\\[.*?\\\]|\$.*?\$|[A-Za-z]\w*\s*\(\w+\s*,\s*\w+\)\s*=\s*[^=]+|[A-Za-z]\w*\s*=\s*[^=]+|MHA\(.*?\)|softmax\(.*?\)|FFN\(.*?\)",
        re.DOTALL
    )
    formulas = formula_pattern.findall(text)
    formula_count = len(formulas)
    if formula_count == 0:
        return 0.0
    word_count = len(text.split())
    def calculate_formula_complexity(formula):
        score = 0.0
        if any(op in formula for op in ['+', '-', '*', '/']):
            score += 0.1
        score += 0.3 * (formula.count(r'\sum') + formula.count(r'\prod') +
                        formula.count(r'\int') + formula.count(r'\partial'))
        if formula.count(r'\sum') > 1 or r'{' in formula:
            score += 0.2
        score += 0.1 * (formula.count(r'\delta') + formula.count(r'\mu') +
                        formula.count(r'_') + formula.count(r'^'))
        if 'softmax' in formula or 'MHA' in formula or 'FFN' in formula:
            score += 0.3
        return min(1.0, score)
    formula_score = 0.0
    for formula in formulas:
        complexity = calculate_formula_complexity(formula)
        formula_score += complexity * 0.6
    base_scaling_factor = 5.0
    normalized_text_length = max(math.log(word_count + 1), 1) * base_scaling_factor
    raw_density = formula_score / normalized_text_length
    scaled_density = math.log(1 + raw_density)
    scaled_density = scaled_density / (1 + scaled_density)
    return scaled_density

def generate_ngrams(tokens, min_n, max_n):
    ngrams = set()
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(' '.join(tokens[i:i + n]))
    return ngrams

def calculate_knowledge_abstractness(text, term_bank):
    tokens = word_tokenize(text.lower())
    ngrams = generate_ngrams(tokens, 1, 3)
    matched_terms = ngrams & term_bank.terms
    total_ngrams = len(ngrams)
    term_density = len(matched_terms) / max(total_ngrams, 1)
    weighted_term_score = 0
    for term in matched_terms:
        n_words = len(term.split())
        if n_words == 1:
            weight = 1.0
        elif n_words == 2:
            weight = 1.5
        else:
            weight = 2.0
        weighted_term_score += weight
    weighted_term_score = weighted_term_score / max(total_ngrams, 20)
    w1, w2 = 0.5, 0.5
    knowledge_abstractness = w1 * term_density + w2 * weighted_term_score
    return min(max(knowledge_abstractness, 0.0), 1.0)

def calculate_structure_disorder(text):
    section_patterns = [
        r'(?i)\b(Chapter|Part|Section|Module|Unit)\s+[\w\d.]+\b',
        r'(?i)\b\d+\.\d+\s+[A-Z][a-z]+\b',
        r'^#{1,3}\s+.+$'
    ]
    section_count = sum(len(re.findall(p, text)) for p in section_patterns)
    word_count = len(text.split())
    length_factor = min(1.0, word_count / 200)
    sentences = sent_tokenize(text)
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    newline_count = text.count("\n")
    jumpiness_score = abs(len(sentences) - avg_sentence_length)
    jumpiness_factor = min(1.0, (jumpiness_score + newline_count) / 15)
    section_factor = 1 / (section_count + 1)
    structure_disorder = (
            section_factor * 0.2 +
            length_factor * 0.3 +
            jumpiness_factor * 0.5
    )
    final_score = round(max(0.1, min(structure_disorder, 1.0)), 2)
    return final_score

# ========================================
# 📥 输入模块
# ========================================

class InputModule:
    def __init__(self, cognitive_load_path, score_path, textbook_path, term_bank_path="academic_terms.txt"):
        self.term_bank = AcademicTermBank(term_bank_path)
        self.cognitive_load_data = self.load_cognitive_load(cognitive_load_path)
        self.score_data = self.load_scores(score_path)
        self.textbook_data = self.load_textbook(textbook_path)

    def load_cognitive_load(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if 'responses' not in df.columns or df['responses'].apply(lambda x: len(x) != 8).any():
                raise ValueError("认知负荷数据格式错误：每个学生必须有 8 个评分")
            df['has_valid_responses'] = df['responses'].apply(lambda x: any(v is not None for v in x))
            if not df['has_valid_responses'].any():
                print("警告：所有认知负荷数据均为 null，使用默认值")
                df['cognitive_load_level'] = 'medium'
                df['cognitive_load_score'] = 50.0
            else:
                df[['cognitive_load_level', 'cognitive_load_score']] = df['responses'].apply(self.calculate_cognitive_load).apply(pd.Series)
            return df
        except Exception as e:
            raise ValueError(f"加载认知负荷数据失败: {str(e)}")

    def calculate_cognitive_load(self, responses):
        valid_responses = [r for r in responses if r is not None]
        if not valid_responses:
            return pd.Series(['medium', 50.0])
        total_score = sum(valid_responses)
        normalized_score = (total_score - len(valid_responses)) / (5 * len(valid_responses) - len(valid_responses)) * 100
        if normalized_score >= 70:
            return pd.Series(['high', normalized_score])
        elif normalized_score >= 40:
            return pd.Series(['medium', normalized_score])
        else:
            return pd.Series(['low', normalized_score])

    def load_scores(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            if 'score' not in df.columns or df['score'].isnull().any():
                raise ValueError("成绩数据格式错误：每个学生必须有有效成绩")
            df[['score_level', 'score']] = df['score'].apply(self.classify_score).apply(pd.Series)
            return df
        except Exception as e:
            raise ValueError(f"加载成绩数据失败: {str(e)}")

    def classify_score(self, score):
        if score >= 85:
            return pd.Series(['high', score])
        elif score >= 50:
            return pd.Series(['medium', score])
        else:
            return pd.Series(['low', score])

    def load_textbook(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("教材数据必须为 JSON 数组")
            return data
        except Exception as e:
            raise ValueError(f"加载教材数据失败: {str(e)}")

# ========================================
# 🧠 评估模块
# ========================================

class EvaluationModule:
    def __init__(self, model_path='best_model_xgb.pkl', scaler_path='scaler.pkl', weights_path='weights_xgb.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.weights_info = joblib.load(weights_path)
        self.term_bank = AcademicTermBank()
        self.table_analyzer = TableComplexityAnalyzer()

    def evaluate_textbook_difficulty(self, textbook_data):
        features = []
        for entry in textbook_data:
            text = entry.get("text", "")
            linguistic_complexity = calculate_language_complexity(text, self.term_bank)
            formula_density = calculate_formula_density(text)
            diagram_complexity = self.table_analyzer.analyze_entry(entry)
            knowledge_abstraction = calculate_knowledge_abstractness(text, self.term_bank)
            structural_disorganization = calculate_structure_disorder(text)
            feature_dict = {
                'linguistic_complexity': linguistic_complexity,
                'formula_density': formula_density,
                'diagram_complexity': diagram_complexity,
                'knowledge_abstraction': knowledge_abstraction,
                'structural_disorganization': structural_disorganization,
                'formula_knowledge_interaction': formula_density * knowledge_abstraction,
                'complex_knowledge_interaction': linguistic_complexity * knowledge_abstraction,
                'has_formula': int(formula_density > 0),
                'has_diagram': int(diagram_complexity > 0),
                'has_structural_disorganization': int(structural_disorganization > 0)
            }
            features.append(feature_dict)
        features_df = pd.DataFrame(features)
        feature_names_model_order = [
            'complex_word_ratio', 'formula_density', 'knowledge_abstractness',
            'structure_disorder', 'table_complexity',
            'formula_knowledge_interaction', 'complex_knowledge_interaction',
            'has_formula', 'has_table', 'has_structure_disorder'
        ]
        feature_name_mapping = {
            'complex_word_ratio': 'linguistic_complexity',
            'knowledge_abstractness': 'knowledge_abstraction',
            'structure_disorder': 'structural_disorganization',
            'table_complexity': 'diagram_complexity',
            'has_table': 'has_diagram',
            'has_structure_disorder': 'has_structural_disorganization'
        }
        for model_feature_name in feature_names_model_order:
            if model_feature_name in feature_name_mapping:
                features_df[model_feature_name] = features_df[feature_name_mapping[model_feature_name]]
        X = features_df[feature_names_model_order].values
        if np.any(np.isnan(X)):
            print("警告：特征包含缺失值，使用中位数填充")
            for i in range(X.shape[1]):
                X[:, i] = np.where(np.isnan(X[:, i]), np.nanmedian(X[:, i]), X[:, i])
        formula_idx_model = feature_names_model_order.index('formula_density')
        X[:, formula_idx_model] = np.log1p(X[:, formula_idx_model])
        table_idx_model = feature_names_model_order.index('table_complexity')
        bins = self.weights_info.get('table_complexity_bins', [0, 0.1, 0.5, 1.0, 2.0])
        cut_result = pd.cut(X[:, table_idx_model], bins=bins, labels=False, include_lowest=True)
        if np.any(np.isnan(X[:, table_idx_model])):
            print(f"警告：在进行 table_complexity 分箱前发现 NaN 值，使用中位数填充")
            original_vals = X[:, table_idx_model]
            median_val = np.nanmedian(original_vals)
            X[:, table_idx_model] = np.where(np.isnan(original_vals), median_val, original_vals)
            cut_result = pd.cut(X[:, table_idx_model], bins=bins, labels=False, include_lowest=True)
        X[:, table_idx_model] = cut_result.astype(np.float64)
        X_std = self.scaler.transform(X)
        difficulty_scores = self.model.predict(X_std)
        all_zero_mask = np.all(X[:, :5] == 0, axis=1)
        difficulty_scores[all_zero_mask] = 1.0
        features_df['difficulty_score'] = np.round(difficulty_scores, 1)
        overall_difficulty = features_df['difficulty_score'].mean()
        return features_df, overall_difficulty

# ========================================
# 🎯 IRT 自适应调节
# ========================================

class IRTOptimizedLearningAdaptation:
    def __init__(self):
        self.D = 1.702
        self.a = 1.7
        self.b_original_min = 1.0
        self.b_original_max = 10.0
        self.b_mapped_min = 1.0
        self.b_mapped_max = 5.0

    def _map_difficulty_scale(self, b_original):
        b_mapped = self.b_mapped_min + \
                   (b_original - self.b_original_min) / (self.b_original_max - self.b_original_min) * \
                   (self.b_mapped_max - self.b_mapped_min)
        return b_mapped

    def calculate_ability(self, score_level, score, cognitive_load_level, cognitive_load_score):
        if score < 50:
            theta_score = 2.0
        elif score >= 85:
            theta_score = 4.5
        else:
            theta_score = 2.0 + (score - 50) / (85 - 50) * (4.5 - 2.0)
        theta_load = 0.0
        if cognitive_load_level == 'high':
            theta_load = -1.0
        elif cognitive_load_level == 'low':
            theta_load = +1.0
        theta = theta_score + theta_load
        return max(min(theta, 5.0), 1.0)

    def calculate_probability(self, theta, b_mapped):
        return 1 / (1 + np.exp(-self.D * self.a * (theta - b_mapped)))

    def adjust_difficulty(self, delta, P_theta):
        if delta >= 2.0 and P_theta < 0.12:
            return 'significant_downgrade', '大幅降低语言复杂度，简化结构'
        elif 1.0 <= delta < 2.0 and 0.12 <= P_theta < 0.27:
            return 'moderate_downgrade', '适度降低语言复杂度，增加示例'
        elif -1.0 <= delta < 1.0 and 0.27 <= P_theta <= 0.73:
            return 'maintain', '优化图表，补充练习'
        elif delta < -1.0 and P_theta >= 0.73:
            return 'upgrade', '增加开放性问题，增强抽象度'
        elif delta >= 1.0 and P_theta >= 0.27 or (-1.0 <= delta < 2.0 and P_theta < 0.12):
            return 'slight_downgrade', '略微降低语言复杂度，增加辅助说明'
        elif delta < -1.0 and P_theta < 0.73 or (-2.0 <= delta < 1.0 and P_theta > 0.73):
            return 'slight_upgrade', '略微增加内容深度，引入简单挑战'
        else:
            return 'slight_adjust', '微调内容结构，保持现有难度'

    def predict_optimal_challenge(self, student_ability, overall_difficulty_original):
        b_mapped = self._map_difficulty_scale(overall_difficulty_original)
        P_theta = self.calculate_probability(student_ability, b_mapped)
        delta = b_mapped - student_ability
        adjustment, suggestion = self.adjust_difficulty(delta, P_theta)
        return {
            'difficulty_score': round(b_mapped, 2),
            'P_theta': round(P_theta, 2),
            'delta': round(delta, 2),
            'adjustment': adjustment,
            'suggestion': suggestion
        }

# ========================================
# 📚 主系统
# ========================================

class TextbookDifficultySystem:
    def __init__(self, cognitive_load_path, score_path, textbook_path, model_path='best_model_xgb.pkl',
                 scaler_path='scaler.pkl', weights_path='weights_xgb.pkl', term_bank_path='academic_terms.txt',
                 minimax_api_key=None):
        self.input_module = InputModule(cognitive_load_path, score_path, textbook_path, term_bank_path)
        self.evaluation_module = EvaluationModule(model_path, scaler_path, weights_path)
        self.irt_olad = IRTOptimizedLearningAdaptation()
        self.minimax_api_key = minimax_api_key or "your_minimax_api_key"

    def generate_new_textbook(self, textbook_snippet, features, theta, delta, adjustment, suggestion):
        prompt = f"""
        你是一个教育机器人，任务是生成与学生能力匹配的个性化教材内容。基于以下信息：
        - 原教材内容：{textbook_snippet}
        - 五维难易度分数：
          - 语言复杂性：{features['linguistic_complexity']}
          - 公式密度：{features['formula_density']}
          - 视觉复杂性：{features['diagram_complexity']}
          - 知识抽象度：{features['knowledge_abstraction']}
          - 结构无序度：{features['structural_disorganization']}
          - 综合难度：{features['difficulty_score']}
        - 学生能力（θ）：{theta}
        - 能力与难度差值（Δ）：{delta}
        - 调整策略：{suggestion}
        生成一个新的教材片段（约100-200字），与原教材主题相关，格式为JSON：
        {{
          "text": "新教材内容",
          "image_path": ""
        }}
        规则：
        - 如果Δ≥2（significant_downgrade）：简化语言（避免术语），移除公式，降低抽象度。
        - 如果1≤Δ<2（moderate_downgrade）：使用简单措辞，增加1-2个示例，减少公式。
        - 如果-1≤Δ<1（maintain）：保持难度，优化结构，添加清晰标题。
        - 如果Δ<-1（upgrade）：增加抽象概念，引入1个简单公式，保持清晰结构。
        - 如果公式密度>0.3，减少公式；如果语言复杂性>0.5，简化措辞；如果知识抽象度>0.4，减少抽象术语。
        """
        headers = {
            "Authorization": f"Bearer {self.minimax_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "abab6.5s-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7
        }
        response = requests.post("https://api.minimax.chat/v1/text/chatcompletion", headers=headers, json=data)
        response_data = response.json()
        try:
            new_textbook = json.loads(response_data['choices'][0]['message']['content'])
        except:
            new_textbook = {"text": response_data['choices'][0]['message']['content'], "image_path": ""}
        return new_textbook

    def run(self, output_path=None):
        student_data = pd.merge(
            self.input_module.cognitive_load_data[['student_id', 'cognitive_load_level', 'cognitive_load_score']],
            self.input_module.score_data[['student_id', 'score_level', 'score']],
            on='student_id'
        )
        textbook_data = self.input_module.textbook_data
        features_df, overall_difficulty_original = self.evaluation_module.evaluate_textbook_difficulty(textbook_data)

        results = []
        new_textbooks = []
        for _, student in student_data.iterrows():
            theta = self.irt_olad.calculate_ability(
                student['score_level'], student['score'],
                student['cognitive_load_level'], student['cognitive_load_score']
            )
            challenge = self.irt_olad.predict_optimal_challenge(theta, overall_difficulty_original)
            challenge['student_id'] = student['student_id']
            challenge['cognitive_load_level'] = student['cognitive_load_level']
            challenge['cognitive_load_score'] = student['cognitive_load_score']
            challenge['score'] = student['score']
            challenge['match_status'] = "匹配" if -1 <= challenge['delta'] <= 1 else ("需降低难度" if challenge['delta'] > 1 else "需提升挑战")
            results.append(challenge)

            new_textbook = self.generate_new_textbook(
                textbook_snippet=textbook_data[0]['text'],
                features=features_df.iloc[0].to_dict(),
                theta=theta,
                delta=challenge['delta'],
                adjustment=challenge['adjustment'],
                suggestion=challenge['suggestion']
            )
            new_textbooks.append({
                'student_id': student['student_id'],
                'new_textbook': new_textbook
            })

        result_df = pd.DataFrame(results)
        result_df = result_df[[
            'student_id', 'cognitive_load_level', 'cognitive_load_score', 'score',
            'difficulty_score', 'delta', 'P_theta', 'adjustment', 'suggestion', 'match_status'
        ]].rename(columns={
            'difficulty_score': '教材难易度',
            'delta': 'Δ范围',
            'P_theta': 'P(θ)区间',
            'adjustment': '调节等级',
            'suggestion': '操作建议',
            'cognitive_load_score': '认知负荷得分',
            'match_status': '匹配状态'
        })

        output_json = {
            "textbook_features": features_df[[
                'difficulty_score', 'linguistic_complexity', 'formula_density',
                'diagram_complexity', 'knowledge_abstraction', 'structural_disorganization'
            ]].to_dict(orient='records'),
            "overall_difficulty": round(overall_difficulty_original, 2),
            "students": result_df.to_dict(orient='records'),
            "new_textbooks": new_textbooks
        }

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'results_{timestamp}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至 {output_path}")

        return output_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="教材难易度预测与调节系统")
    parser.add_argument('--cognitive_load', type=str, default='cognitive_load.json', help="认知负荷数据 JSON 文件路径")
    parser.add_argument('--scores', type=str, default='scores.json', help="成绩数据 JSON 文件路径")
    parser.add_argument('--textbook', type=str, default='textbook.json', help="教材数据 JSON 文件路径")
    parser.add_argument('--output', type=str, default='results.json', help="输出 JSON 文件路径")
    parser.add_argument('--model', type=str, default='best_model_xgb.pkl', help="XGBoost模型路径")
    parser.add_argument('--scaler', type=str, default='scaler.pkl', help="标准化器路径")
    parser.add_argument('--weights', type=str, default='weights_xgb.pkl', help="权重文件路径")
    parser.add_argument('--term_bank', type=str, default='academic_terms.txt', help="术语库文件路径")
    parser.add_argument('--minimax_api_key', type=str, required=True, help="MiniMax API 密钥")
    args = parser.parse_args()

    system = TextbookDifficultySystem(
        cognitive_load_path=args.cognitive_load,
        score_path=args.scores,
        textbook_path=args.textbook,
        model_path=args.model,
        scaler_path=args.scaler,
        weights_path=args.weights,
        term_bank_path=args.term_bank,
        minimax_api_key=args.minimax_api_key
    )
    result = system.run(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
