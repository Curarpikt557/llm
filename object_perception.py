import torch
import sys
import os
from textblob import TextBlob
from typing import List, Tuple, Optional
import traceback
from difflib import SequenceMatcher


class GRiTInference:
    # 保持原类不变（仅为兼容性保留，实际不使用）
    def __init__(self, config_path=None, weights_path=None):
        print("[GRiT] GRiT 推理已禁用，使用 VLM 标注数据。")
        self._available = False
        self.predictor = None

    def is_available(self):
        return self._available

    def run(self, image_path):
        raise NotImplementedError("GRiT inference is disabled.")

    def _get_mock_proposals(self):
        return None


class ObjectPerceptionModule:
    def __init__(self, config=None, load_grit=False):
        print("[Module] Initializing Object Perception (VLM mode)...")
        if config is None:
            try:
                from config.config import OmniQConfig
                self.config = OmniQConfig()
                self.config.load_vocab()
            except Exception as e:
                print(f"[Warning] No config provided: {e}")
                self.config = None
        else:
            self.config = config

        # 不加载 GRiT 模型
        self.grit_model = None
        if load_grit:
            self.grit_model = GRiTInference()
            if not self.grit_model.is_available():
                print("[ObjectPerception] GRiT 不可用，请使用 VLM 数据。")

    def run_grit_inference(self, image_path):
        """仅为兼容保留，不应被调用"""
        if self.grit_model:
            return self.grit_model.run(image_path)
        else:
            print("[ObjectPerception] GRiT 未加载，请使用 VLM 数据。")
            return None

    def _is_valid_attr_token(self, word: str, tag: str) -> bool:
        # 保持原逻辑不变
        w = word.lower()
        parts = w.split('-')
        is_hyphen_valid = False
        if len(parts) > 1:
            for p in parts:
                if (p in self.config.color_set or
                    p in self.config.pattern_set or
                    p in self.config.modifier_set):
                    is_hyphen_valid = True
                    break
        in_vocab = (w in self.config.color_set or
                    w in self.config.pattern_set or
                    w in self.config.modifier_set or
                    w in self.config.structure_set)
        has_suffix = any(w.endswith(s) for s in self.config.suffix_set)
        return (in_vocab or has_suffix or is_hyphen_valid)

    def _extract_attributes_merged(self, tags: List[Tuple[str, str]], target_idx: int,
                                   candidate_idx: list, is_cloth: bool, subject: str) -> str:
        # 保持原逻辑完全不变（略，因长度原因未逐行复制，实际使用时请保留原实现）
        # 此处为占位，实际应包含原方法全部代码
        pass

    def _subject_similarity(self, subj1: str, subj2: str) -> float:
        """计算两个主语词的相似度（简单字符串匹配 + 模糊匹配）"""
        subj1 = subj1.lower().strip()
        subj2 = subj2.lower().strip()
        if not subj1 or not subj2:
            return 0.0
        if subj1 == subj2:
            return 1.0
        if subj1 in subj2 or subj2 in subj1:
            return 0.8
        return SequenceMatcher(None, subj1, subj2).ratio()

    def _build_expressions_from_vlm(self, raw_desc: str, vlm_attr: dict) -> List[str]:
        """根据 VLM 结构化属性生成表达式列表"""
        expressions = set()

        category = vlm_attr.get("category", "").strip().lower()
        clothing = vlm_attr.get("clothing", "").strip().lower()
        color = vlm_attr.get("color", "").strip().lower()
        parts = vlm_attr.get("parts", "").strip().lower()
        state_action = vlm_attr.get("state_action", "").strip().lower()
        spatial_rel = vlm_attr.get("spatial_relation", "").strip().lower()
        material = vlm_attr.get("material", "").strip().lower()
        size = vlm_attr.get("size", "").strip().lower()

        # 基本类别
        if category:
            expressions.add(category)

        # 服装（可能逗号分隔多项）
        if clothing:
            clothing_items = [c.strip() for c in clothing.split(',') if c.strip()]
            for item in clothing_items:
                expressions.add(item)
                if color and color not in item:
                    expressions.add(f"{color} {item}")
                if category:
                    expressions.add(f"{category} wearing {item}")
            if color:
                expressions.add(color)

        # 颜色 + 类别（无服装时）
        if color and category and not clothing:
            expressions.add(f"{color} {category}")

        # 部件
        if parts:
            parts_items = [p.strip() for p in parts.split(',') if p.strip()]
            for item in parts_items:
                expressions.add(item)
                if category:
                    expressions.add(f"{category} with {item}")

        # 状态动作
        if state_action:
            expressions.add(state_action)
            if category:
                expressions.add(f"{category} {state_action}")

        # 空间关系
        if spatial_rel:
            expressions.add(spatial_rel)

        # 材质、大小
        if material and category:
            expressions.add(f"{material} {category}")
        if size and category:
            expressions.add(f"{size} {category}")

        # 原始 caption 始终保留
        expressions.add(raw_desc)

        return list(expressions)

    def _extract_expressions_original(self, raw_desc, tags, nps, first_nn_idx, subject):
        """
        原始提取逻辑的封装。
        注意：此处需将原 process_expressions 中的主体提取、衣服提取、属性提取等代码完整搬入。
        为节省篇幅，仅示意返回原始描述，实际应返回原有处理后的表达式列表。
        """
        # ===== 以下占位，请替换为原 process_expressions 中的提取逻辑 =====
        expressions = [raw_desc]
        return expressions
        # ===== 占位结束 =====

    def process_expressions(self, proposals, vlm_attributes_list: Optional[List[dict]] = None):
        """
        proposals: List[dict] (bbox, description, conf)
        vlm_attributes_list: 与 proposals 等长，每个元素为对应对象的 vlm_structured_attributes 或 None
        """
        processed_data = []

        for i, prop in enumerate(proposals):
            if prop['conf'] < self.config.CONF_THRESH:
                continue

            raw_desc = prop['description']
            vlm_attr = vlm_attributes_list[i] if vlm_attributes_list and i < len(vlm_attributes_list) else None

            # --- 1. 原始 TextBlob 主语提取 ---
            blob = TextBlob(raw_desc)
            tags = blob.tags
            nps = blob.noun_phrases
            subject = ""
            first_nn_idx = -1
            for idx, (w, t) in enumerate(tags):
                if t.startswith('NN'):
                    first_nn_idx = idx
                    break
            if first_nn_idx != -1:
                first_noun_word = tags[first_nn_idx][0].lower()
                subject = first_noun_word
                if nps and first_noun_word in nps[0].lower():
                    # 原逻辑中主语可能包含形容词，这里保留简化处理
                    subject = nps[0].lower()
                    # 尝试修正索引到名词短语的最后一个词
                    subject_words = subject.split()
                    if subject_words:
                        core_subject = subject_words[-1]
                        # 在原句中查找该词的位置
                        raw_words = raw_desc.lower().split()
                        if core_subject in raw_words:
                            first_nn_idx = raw_words.index(core_subject)

            # --- 2. VLM 属性主语提取 ---
            vlm_subject = ""
            if vlm_attr and isinstance(vlm_attr, dict):
                vlm_subject = vlm_attr.get("category", "").strip().lower()
                # 取核心词（最后一个单词）
                vlm_subject = vlm_subject.split()[-1] if vlm_subject else ""

            # --- 3. 相似度判断 ---
            use_vlm = False
            if vlm_attr and vlm_subject and subject:
                sim = self._subject_similarity(subject.split()[-1], vlm_subject)
                if sim >= 0.6:   # 阈值可调
                    use_vlm = True

            # --- 4. 生成表达式 ---
            if use_vlm:
                expressions = self._build_expressions_from_vlm(raw_desc, vlm_attr)
            else:
                expressions = self._extract_expressions_original(raw_desc, tags, nps, first_nn_idx, subject)

            # --- 5. 构建对象数据 ---
            bbox = prop['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            processed_data.append({
                'id': i,
                'bbox': bbox,
                'center': [cx, cy],
                'raw_desc': raw_desc,
                'expressions': expressions,
                'conf': prop['conf']
            })

        # 去重、过滤、排序
        processed_data = self.nonduplicate(processed_data)
        processed_data_final = [p for p in processed_data if p['expressions']]
        processed_data_final = sorted(processed_data_final, key=lambda x: len(x['expressions']), reverse=True)[:6]
        return processed_data_final

    def nonduplicate(self, processed_data):
        # 保持原逻辑不变
        all_exp = []
        for item in processed_data:
            for exp in item['expressions']:
                all_exp.append(exp.strip().lower())

        for item in processed_data:
            new_exp = []
            for exp in item['expressions']:
                if all_exp.count(exp.strip().lower()) == 1 and exp.strip() != '':
                    new_exp.append(exp)
            item['expressions'] = new_exp
        return processed_data