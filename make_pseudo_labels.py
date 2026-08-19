import json
import os
import sys
import torch
from collections import Counter, defaultdict
from tqdm import tqdm

# 可选导入 TextBlob
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False


def load_full_attr(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_label(s: str) -> str:
    return ' '.join(s.lower().split()) if s else ''


def extract_category(obj):
    """提取类别，优先 category，缺失时用 TextBlob 提取名词短语"""
    attrs = obj.get('vlm_structured_attributes', {}) or {}
    category = attrs.get('category')
    if category:
        return normalize_label(category)

    raw_class = obj.get('class') or ''
    if raw_class:
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(raw_class)
                phrases = blob.noun_phrases
                if phrases:
                    return normalize_label(phrases[0])
            except Exception:
                pass
        return normalize_label(raw_class.split()[0])
    return ''


def get_attrs(obj):
    """提取对象属性（忽略 parts）"""
    attrs = obj.get('vlm_structured_attributes', {}) or {}
    return {
        'category': extract_category(obj),
        'color': normalize_label(attrs.get('color') or ''),
        'size': normalize_label(attrs.get('size') or ''),
        'material': normalize_label(attrs.get('material') or ''),
        'clothing': normalize_label(attrs.get('clothing') or ''),
        'state': normalize_label(attrs.get('state_action') or attrs.get('state') or ''),
    }


def ordinal(n):
    """将整数转换为英文序数词，如 1->first, 2->second"""
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return str(n) + suffix


def generate_ordinal_spatial_description(obj, objects):
    """
    根据对象的 2D x 坐标（从左到右）生成唯一序号描述。
    返回类似 "the first {category} from the left" 的字符串。
    """
    category = get_attrs(obj)['category']
    if not category:
        category = "object"

    same_cat = [o for o in objects if get_attrs(o)['category'] == category]
    same_cat_sorted = sorted(same_cat, key=lambda o: (o.get('center_2d', [0, 0])[0], o.get('center_2d', [0, 0])[1]))

    try:
        rank = same_cat_sorted.index(obj)
    except ValueError:
        rank = 0

    n = len(same_cat_sorted)
    if n <= 1:
        return f"the only {category}"

    if rank == 0:
        return f"the first {category} from the left"
    elif rank == n - 1:
        return f"the rightmost {category}"
    else:
        return f"the {ordinal(rank+1)} {category} from the left"


def compute_spatial_relation(group_objects, idx):
    """
    基于同类别对象列表计算单一空间标签。
    优先返回显著的前后关系（front/back），否则返回水平标签。
    避免组合词如 right-middle、middle-center。
    """
    n = len(group_objects)
    if n == 1:
        return "center"

    obj = group_objects[idx]

    # 水平方向：按 x 坐标排序
    xs = [o.get('center_2d', [0, 0])[0] for o in group_objects]
    sorted_indices = sorted(range(n), key=lambda k: xs[k])
    rank = sorted_indices.index(idx)

    if n == 2:
        horiz = 'leftmost' if rank == 0 else 'rightmost'
    elif n == 3:
        horiz = 'leftmost' if rank == 0 else ('middle' if rank == 1 else 'rightmost')
    elif n == 4:
        horiz = ['leftmost', 'left', 'right', 'rightmost'][rank]
    elif n == 5:
        horiz = ['leftmost', 'left', 'middle', 'right', 'rightmost'][rank]
    else:
        if rank == 0:
            horiz = 'leftmost'
        elif rank == n - 1:
            horiz = 'rightmost'
        else:
            pos = rank / (n - 1)
            if pos < 0.2:
                horiz = 'left'
            elif pos < 0.4:
                horiz = 'middle'
            elif pos < 0.6:
                horiz = 'middle'
            elif pos < 0.8:
                horiz = 'right'
            else:
                horiz = 'right'

    # 深度方向：如果所有对象都有 center_3d 第三维（z），则计算前后关系
    depth_vals = []
    for o in group_objects:
        c3 = o.get('center_3d')
        if c3 and len(c3) >= 3:
            depth_vals.append(c3[2])
        else:
            depth_vals.append(None)

    if all(d is not None for d in depth_vals):
        # 假设 z 值越大越靠前（可根据实际数据调整）
        sorted_depth_indices = sorted(range(n), key=lambda k: depth_vals[k], reverse=True)
        depth_rank = sorted_depth_indices.index(idx)
        if depth_rank == 0:
            depth = 'front'
        elif depth_rank == n - 1:
            depth = 'back'
        else:
            depth = 'middle'
    else:
        depth = None

    # 如果存在显著的前后关系（front/back），优先返回深度标签
    if depth in ('front', 'back'):
        return depth
    # 否则返回水平标签
    return horiz


def matches_condition(obj, condition):
    """判断对象是否满足条件（不含空间）"""
    attrs = get_attrs(obj)
    if condition.get('category') and attrs['category'] != condition['category']:
        return False
    for attr in ['color', 'clothing', 'size', 'material', 'state']:
        if condition.get(attr) and attrs.get(attr) != condition[attr]:
            return False
    return True


def is_unique_condition(condition, objects):
    """检查条件是否唯一指代一个对象"""
    matched = []
    for o in objects:
        if matches_condition(o, condition):
            if condition.get('spatial'):
                if o.get('_spatial_token', '') == condition['spatial']:
                    matched.append(o)
            else:
                matched.append(o)
    return len(matched) == 1


def condition_to_expressions(condition, attrs, spatial_token=None):
    """将唯一条件转换为自然语言表达式（1-2 个变体）"""
    category = condition.get('category', '')
    color = condition.get('color', '')
    size = condition.get('size', '')
    material = condition.get('material', '')
    clothing = condition.get('clothing', '')
    state = condition.get('state', '')
    spatial = condition.get('spatial', '')

    adjectives = []
    if size:
        adjectives.append(size)
    if color:
        adjectives.append(color)
    if material:
        adjectives.append(material)
    base = ' '.join(adjectives + [category]).strip() if (adjectives or category) else category

    exprs = []
    if base:
        exprs.append(base)

    # 后置修饰
    post_parts = []
    if clothing:
        post_parts.append(f"wearing {clothing}")
    if state:
        post_parts.append(f"that is {state}")
    if post_parts:
        exprs.append(base + ' ' + ' '.join(post_parts))

    # 空间表达（只使用单一空间标签，避免组合）
    if spatial:
        if spatial in ('leftmost', 'rightmost'):
            exprs.append(f"the {spatial} {base}")
            exprs.append(f"{base} on the far {'left' if spatial == 'leftmost' else 'right'}")
        elif spatial in ('left', 'right'):
            exprs.append(f"{base} on the {spatial}")
            exprs.append(f"the {spatial} {base}")
        elif spatial in ('middle', 'center'):
            exprs.append(f"{base} in the middle")
            exprs.append(f"the middle {base}")
        elif spatial == 'front':
            exprs.append(f"{base} in the front")
            exprs.append(f"the front {base}")
        elif spatial == 'back':
            exprs.append(f"{base} at the back")
            exprs.append(f"the back {base}")
        else:
            # 其他情况直接前置（通常不会出现）
            exprs.append(f"{spatial} {base}")

    return [normalize_label(e) for e in exprs if e]


def build_unique_label(obj, objects, category_counts):
    """为单个对象生成候选描述（1-2 个，基于单个属性组合）"""
    attrs = get_attrs(obj)
    category = attrs['category']

    # 类别唯一：直接返回类别名
    if category and category_counts.get(category, 0) == 1:
        obj['_primary_label'] = category
        return [category]

    spatial_token = obj.get('_spatial_token', '')

    # 单个属性列表（按优先级），忽略 parts
    attr_order = ['color', 'clothing', 'size', 'material', 'state', 'spatial']
    for attr in attr_order:
        # 如果属性为空则跳过（空间始终可用）
        if attr != 'spatial' and not attrs.get(attr):
            continue

        cond = {'category': category}
        if attr == 'spatial':
            cond['spatial'] = spatial_token
        else:
            cond[attr] = attrs[attr]

        if is_unique_condition(cond, objects):
            exprs = condition_to_expressions(cond, attrs, spatial_token)
            # 去重
            seen = set()
            exprs_clean = []
            for e in exprs:
                if e and e not in seen:
                    seen.add(e)
                    exprs_clean.append(e)
            # 按长度排序，取最短的 1-2 个
            exprs_clean.sort(key=len)
            selected = exprs_clean[:2]
            obj['_primary_label'] = selected[0]
            return selected

    # 所有单个属性组合均不唯一，使用序号描述兜底
    fallback = generate_ordinal_spatial_description(obj, objects)
    obj['_primary_label'] = fallback
    return [fallback]


def build_pseudo_labels(data):
    """从 JSON 数据生成伪标签，不依赖外部模型"""
    all_queries = []

    for img in tqdm(data, desc="Building pseudo labels"):
        image_id = img.get('image_id')
        objects = img.get('objects_3d', [])

        for i, o in enumerate(objects):
            o['_index'] = i

        # 提取类别并处理空类别
        cats = []
        for o in objects:
            cat = get_attrs(o)['category']
            if not cat:
                cat = f"unknown_{o['_index']}"
            cats.append(cat)
        cat_counts = Counter(cats)

        # 按类别分组，计算同类别空间 token
        category_to_objects = defaultdict(list)
        for o in objects:
            cat = get_attrs(o)['category']
            if not cat:
                cat = f"unknown_{o['_index']}"
            category_to_objects[cat].append(o)

        for cat, group in category_to_objects.items():
            for idx_in_group, o in enumerate(group):
                o['_spatial_token'] = compute_spatial_relation(group, idx_in_group)

        # 生成候选描述
        obj_exprs = {}
        for o in objects:
            expr_list = build_unique_label(o, objects, cat_counts)
            obj_exprs[o['_index']] = expr_list

        # 描述字符串全局去重
        desc_to_objects = defaultdict(list)
        for o in objects:
            for desc in obj_exprs[o['_index']]:
                desc_to_objects[desc].append(o['_index'])

        conflict_descs = {desc for desc, idxs in desc_to_objects.items() if len(idxs) > 1}

        for o in objects:
            idx = o['_index']
            cleaned_exprs = [d for d in obj_exprs[idx] if d not in conflict_descs]
            if not cleaned_exprs:
                # 若全部冲突，使用序号描述生成唯一描述
                cleaned_exprs = [generate_ordinal_spatial_description(o, objects)]
            obj_exprs[idx] = cleaned_exprs

        # 输出每条描述为独立样本
        for o in objects:
            bbox = o.get('bbox')
            for expr in obj_exprs[o['_index']]:
                all_queries.append([
                    f"{image_id}.jpg",
                    '_',
                    bbox,
                    expr,
                    '_'
                ])

    return all_queries


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    json_path = os.path.join(base, 'full_attr.json')
    out_pth = os.path.join(base, '../../omniq/11.pth')

    print(f"Loading attributes from: {json_path}")
    data = load_full_attr(json_path)
    queries = build_pseudo_labels(data)
    print(f"Built {len(queries)} pseudo label entries. Saving to {out_pth}")
    torch.save(queries, out_pth)
    print("Saved.")