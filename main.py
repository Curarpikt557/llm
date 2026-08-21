import os
import sys
import json
import argparse
import glob
import cv2
from tqdm import tqdm
import numpy as np
import torch

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "MiDaS"))

from config.config import OmniQConfig
from core.object_perception import ObjectPerceptionModule
from core.spatial_relation import SpatialRelationModule3D
from core.spatial_graph import SpatialGraphModule


def parse_args():
    parser = argparse.ArgumentParser(description="Omni-Q Data Generation Pipeline (VLM mode)")
    parser.add_argument("--vlm_json", type=str, default='../attr/full_attr.json',
                        help="Path to VLM annotated JSON (e.g., grit_caption_refined.json)")
    parser.add_argument("--image_dir", type=str,
                        default="../unc_train",
                        help="Root folder of images")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save generated data")
    parser.add_argument("--midas_weights", type=str,
                        default="./MiDaS/weights/dpt_beit_large_512.pt",
                        help="Path to MiDaS weights")
    parser.add_argument("--midas_type", type=str, default="dpt_beit_large_512",
                        help="MiDaS model type")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化配置
    config = OmniQConfig()
    print("=== Initializing Omni-Q Modules (VLM mode) ===")

    # 初始化模块（不加载 GRiT）
    opm = ObjectPerceptionModule(config, load_grit=False)

    # 初始化空间关系模块
    srm = SpatialRelationModule3D(
        config,
        model_type=args.midas_type,
        weights_path=args.midas_weights
    )

    # 初始化空间图模块（用于生成表达式）
    sgm = SpatialGraphModule()

    # 读取 VLM 标注 JSON
    with open(args.vlm_json, 'r', encoding='utf-8') as f:
        vlm_data = json.load(f)  # 若数据量大，可考虑分批处理
    print(f"Loaded {len(vlm_data)} images from VLM JSON.")

    final_dataset = []
    processed_count = 0

    for img_item in tqdm(vlm_data, desc="Processing images"):
        image_id = img_item["image_id"]
        image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"[Warning] Image not found: {image_path}")
            continue

        objects_3d = img_item.get("objects_3d", [])
        if not objects_3d:
            continue

        # 构建 proposals 和 vlm_attributes_list
        proposals = []
        vlm_attrs = []
        for obj in objects_3d:
            bbox_xywh = obj["bbox"]  # [x, y, w, h]
            x1, y1, w, h = bbox_xywh
            proposals.append({
                'bbox': [x1, y1, w,  h],  # 转为 [x1, y1, x2, y2]
                'description': obj["caption"],
                'conf': obj.get("conf", 0.7)       # 使用 JSON 中的置信度或默认值
            })
            vlm_attrs.append(obj.get("vlm_structured_attributes", None))

        # 处理表达式
        objects = opm.process_expressions(proposals, vlm_attributes_list=vlm_attrs)
        if not objects:
            continue

        # 深度估计
        try:
            # 直接使用绝对路径调用，无需 chdir
            depth_map = srm.estimate_depth(image_path)
        except Exception as e:
            print(f"[Depth Error] {image_id}: {e}")
            continue

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        for obj in objects:
            obj['shape'] = (height, width)
            obj['image_id'] = image_id

        # 计算空间关系
        relations = srm.compute_spatial_relations(objects, depth_map, (height, width))

        # 构建并遍历图，生成最终查询
        queries = sgm.build_and_traverse(objects, relations)

        for q in queries:
            q[0] = q[0] + '.jpg'   # 假设原始为 image_id
            final_dataset.append(q)

        processed_count += 1

    # 保存结果
   
    output_pth = os.path.join(args.output_dir, "../unc_train_pseudo.pth")
    
    torch.save(final_dataset, output_pth)
    print(f"Done! Processed {processed_count} images, generated {len(final_dataset)} queries.")
    print(f"Saved to {output_pth}")


if __name__ == "__main__":
    main()