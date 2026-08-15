import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置 =================
INPUT_JSON = "../attr/full_attr.json"   # 输入文件（包含 vlm_structured_attributes）
OUTPUT_JSON = "./pseudo_labels.json"        # 输出文件（TransVG 格式标签）
LLM_MODEL = "../Qwen3-9B"       # 本地模型路径或 HuggingFace ID
BATCH_SIZE = 8                               # 批处理图片数量
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0

# ================= 加载模型 =================
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()
print("LLM loaded.")

# ================= 系统提示词 =================
SYSTEM_PROMPT = """You are a referring expression generation engine. 
You receive a list of objects from one image, each described by a set of structured attributes. 
Your task is to generate a unique, minimal, and natural English referring expression (label) for each object, so that it can be unambiguously identified within that image.

Rules:
1. Uniqueness: Every label must uniquely identify its target among all objects in the same image. If multiple objects have the same category, use distinguishing attributes.
2. Minimality: Follow Grice's Maxim of Quantity. Only include attributes that are necessary to differentiate the target from other objects of the same category. Do not add redundant information. If the object is the only one of its category, the label can be just the category itself (e.g., "giraffe").
3. Semantic equivalence: Treat synonyms and paraphrases as equivalent (e.g., "blue shirt" and "blue top" are the same distinguishing feature). Do not consider them different.
4. Attribute priority: When disambiguation is needed, add attributes in this order: 
   - For humans: clothing, parts (hair, glasses, etc.), state_action, then spatial_relation.
   - For non-human animals: color, size, parts, state_action, then spatial_relation.
   - For non-living objects: color, material, size, parts, state_action, then spatial_relation.
   Only use spatial_relation if all intrinsic attributes fail to uniquely identify the object.
5. Hallucination prevention: Only use the attribute values provided in the input. Do not invent new attributes or words. If no combination of provided attributes can uniquely identify an object, use a generic fallback: "the [category] #" with an index number (e.g., "the woman #1", "the woman #2").
6. Output format: Return a JSON array of objects, each with "id" (integer) and "label" (string). Do not include any additional text.

Example input:
[
  {"id": 0, "category": "woman", "clothing": "blue shirt", "parts": "glasses"},
  {"id": 1, "category": "woman", "clothing": "blue top", "parts": "scarf"},
  {"id": 2, "category": "man", "clothing": "red jacket"}
]
Example output:
[
  {"id": 0, "label": "the woman in blue shirt with glasses"},
  {"id": 1, "label": "the woman in blue top with scarf"},
  {"id": 2, "label": "the man in red jacket"}
]
"""

# ================= 工具函数 =================
def build_user_prompt(objects_list):
    """构建单个图片的 user prompt"""
    return f"Here are the objects for the current image:\n{json.dumps(objects_list, ensure_ascii=False)}\n\nGenerate unique minimal referring expressions for each object. Output the JSON array now."

def extract_json_array(text):
    """从 LLM 输出中提取 JSON 数组"""
    try:
        return json.loads(text)
    except:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

def post_process_label(label, attrs):
    """
    简单后处理：检查标签中的实词是否在属性字典中出现过。
    若出现未知词（如幻觉），回退到通用标签 "the {category}"。
    """
    words = re.findall(r'[a-zA-Z]+', label.lower())
    stopwords = {"the", "a", "an", "of", "in", "with", "and", "or", "to", "on", "at", "by"}
    unknown = [w for w in words if w not in stopwords and w not in str(attrs).lower()]
    if unknown:
        category = attrs.get("category", "object")
        return f"the {category}"
    return label

def ensure_unique_labels(label_map):
    """如果同一图像内标签重复，则在后续重复项后追加编号 #2, #3..."""
    seen = {}
    unique_map = {}
    for idx, label in label_map.items():
        if label in seen:
            seen[label] += 1
            unique_map[idx] = f"{label} #{seen[label]}"
        else:
            seen[label] = 1
            unique_map[idx] = label
    return unique_map

# ================= 主流程 =================
def main():
    # 读取输入数据
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建待处理的图像列表
    image_groups = []
    for img_item in data:
        objects = img_item.get("objects_3d", [])
        if not objects:
            continue
        obj_list = []
        for idx, obj in enumerate(objects):
            attrs = obj.get("vlm_structured_attributes", {})
            if attrs and "category" in attrs:
                obj_list.append({"id": idx, **attrs})
        if obj_list:
            image_groups.append((img_item, obj_list))

    print(f"Total image groups to process: {len(image_groups)}")

    # 存储最终输出（TransVG 格式）
    transvg_labels = []

    # 批量处理
    for i in tqdm(range(0, len(image_groups), BATCH_SIZE), desc="Generating labels"):
        batch = image_groups[i:i+BATCH_SIZE]
        prompts = [build_user_prompt(obj_list) for _, obj_list in batch]

        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        inputs = tokenizer(chat_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for (img_item, obj_list), gen_text in zip(batch, generated_texts):
            arr = extract_json_array(gen_text)
            if arr is None:
                print(f"Warning: Failed to parse LLM output for image {img_item['image_id']}, using fallback labels.")
                arr = [{"id": obj["id"], "label": f"the {obj.get('category','object')}"} for obj in obj_list]

            # 后处理验证
            label_map = {}
            for item in arr:
                obj_id = item.get("id")
                label = item.get("label", "")
                if obj_id is not None and obj_id < len(obj_list):
                    attrs = obj_list[obj_id]
                    label = post_process_label(label, attrs)
                    label_map[obj_id] = label

            # 确保同一图像内标签唯一
            label_map = ensure_unique_labels(label_map)

            # === 新增：打印当前图片的最终标签，方便查看 ===
            print(f"\n===== Image: {img_item['image_id']} =====")
            for obj_id, label in label_map.items():
                print(f"  Object {obj_id}: {label}")
            print("=" * 40)

            # 构建该图像的所有物体输出记录
            for obj_id, label in label_map.items():
                obj = img_item["objects_3d"][obj_id]
                image_id = img_item.get("image_id", "")
                shape = obj.get("shape", None)      # 从对象中获取 shape，若无则 None
                bbox = obj.get("bbox", None)        # 从对象中获取 bbox，若无则 None
                transvg_labels.append([
                    image_id,
                    shape,
                    bbox,
                    label,
                    'useless placeholder'
                ])

    # 保存最终输出
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(transvg_labels, f, indent=4, ensure_ascii=False)
    print(f"Done. Saved {len(transvg_labels)} labels to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()