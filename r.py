import json
import re
import time
from llama_cpp import Llama

# ======================== 配置 ========================
MODEL_PATH = "qwen2.5-14b-instruct-q5_k_m.gguf"  # "qwen2.5-14b-instruct-q5_k_m.gguf"   #"qwen2.5-7b-instruct-q4_k_m.gguf"
INPUT_JSON = "grit_f.json"           # 输入 JSON 文件
OUTPUT_COCO = "coco_output.json"     # 输出标准的 COCO JSON 结果文件
PROMPT_FILE = "systemm.txt"           # 系统提示文件


MAX_OBJS = 20                        # 每张图最多输入的物体数
MAX_TOKENS = 1024                    # 足够容纳 thought_process + JSON（实测 400-800 tokens）
TEMPERATURE = 0.1
CONTEXT_SIZE = 4096
N_THREADS = 8
MAX_EXPRESSIONS = 6                  # 每张图最多保留表达数，与系统提示一致
# ======================== 加载模型 ========================
print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=CONTEXT_SIZE,
    n_threads=N_THREADS,
    n_batch=512,
    verbose=False
)
print("Model loaded.\n")

# ======================== 读取系统提示 ========================
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    system_prompt = f.read()

# ======================== 读取输入数据 ========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    entries = json.load(f)

# ======================== 构建 prompt ========================
def build_batch_prompt(objects, img_w, img_h):
    obj_lines = []
    for obj in objects:
        obj_lines.append(
            f'id:{obj["id"]} | caption: "{obj["expression"]}" | '
            f'cx:{obj["center_cx"]} cy:{obj["center_cy"]} depth:{obj["depth"]} '
            f'w:{obj["width"]} h:{obj["height"]}'
        )
    objects_str = "\n".join(obj_lines)

    user = (
        f"Image size: {img_w} x {img_h}\n"
        f"Objects:\n{objects_str}\n\n"
        "Select up to 6 objects that can be uniquely referred to, and generate their expressions following the cognitive rules."
    )
    return user

# ======================== 解析 JSON 输出 ========================
def parse_json_output(text):
    """从 LLM 输出中提取 JSON 数组"""
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return []
    try:
        return json.loads(json_str)
    except:
        return []

# ======================== 初始化 COCO 结构 ========================
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "object"}  # 泛化的视觉定位目标大类
    ]
}

annotation_id_counter = 1

# ======================== 主处理流程 ========================
for entry in entries:
    img_id = entry["image_id"]
    img_w = entry["img_w"]
    img_h = entry["img_h"]
    objects = entry["objects"][:MAX_OBJS]

    print(f"Processing Image ID: {img_id} ({img_w}x{img_h}), objects pool: {len(objects)}")
    t_start = time.time()

    # 1. 填充 COCO images 字段
    coco_data["images"].append({
        "id": img_id,
        "width": img_w,
        "height": img_h,
        "file_name": f"{img_id}.jpg"
    })

    # 构建模型输入
    user_prompt = build_batch_prompt(objects, img_w, img_h)
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 模型推理
    output = llm.create_completion(
        prompt=full_prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=["<|im_end|>"]
    )
    raw_output = output["choices"][0]["text"]

    # 解析模型输出的 JSON 数组
    model_expressions = parse_json_output(raw_output)

    # 确保唯一性、不重复且数量受限
    seen_expr = set()
    kept_count = 0

    for item in model_expressions:
        obj_id = item.get("id")
        expr = item.get("expression", "").strip()
        thought = item.get("thought_process", "").strip()

        if expr and expr not in seen_expr:
            seen_expr.add(expr)
            
            # 从原始输入中找回对应的 bounding box
            bbox = next((o["bbox"] for o in objects if o["id"] == obj_id), None)
            original_caption = next((o["expression"] for o in objects if o["id"] == obj_id), "")

            # 2. 填充 COCO annotations 字段
            coco_data["annotations"].append({
                "id": annotation_id_counter,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,                     # [x, y, width, height]
                "expression": expr,               # 优化后的唯一指代标签
                "original_caption": original_caption, # 原始简短标签对比
                "thought_process": thought        # 显式思考过程，方便后续过滤修改
            })
            
            annotation_id_counter += 1
            kept_count += 1
            
            if kept_count >= MAX_EXPRESSIONS:
                break
            print('#'*15)
            print(coco_data["annotations"][-1]["thought_process"])
            print(coco_data["annotations"][-1]["bbox"][2]-coco_data["annotations"][-1]["bbox"][0])
            print(coco_data["annotations"][-1]["expression"])
            print('-'*15)
            print(coco_data["annotations"][-1]["original_caption"])
            print('#'*15)

    t_elapsed = time.time() - t_start
    
    print(f"  Done in {t_elapsed:.2f}s | Successfully annotated {kept_count} unique objects.")
    print('\n')

# ======================== 保存标准 COCO 文件 ========================
with open(OUTPUT_COCO, "w", encoding="utf-8") as f_coco:
    json.dump(coco_data, f_coco, indent=4, ensure_ascii=False)

print(f"\nAll targets converted successfully! Saved in strict COCO format to: {OUTPUT_COCO}")