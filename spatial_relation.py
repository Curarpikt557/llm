import cv2
import numpy as np
import torch
import os
import sys
import glob
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "MiDaS"))
from midas.model_loader import default_models, load_model
from midas import utils
class SpatialRelationModule3D:
    def __init__(self, config=None, model_type="dpt_beit_large_512", 
                 weights_path="/home/user/ljr/omniq/MiDaS/weights/dpt_beit_large_512.pt"):
        print("[SpatialRelation] Initializing 3D Spatial Relation Module (MiDaS)...")
        self.model_type = model_type
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, net_w, net_h = load_model(
            self.device,
            self.weights_path,
            self.model_type,
            optimize=False,
            height=None,
            square=False
        )
        

        # 配置加载
        if config is None:
            try:
                #os.chdir(os.path.join(os.path.dirname(os.getcwd())))
                sys.path.insert(0, os.path.join(os.getcwd(), 'omniq','config'))
                from config import OmniQConfig
                self.config = OmniQConfig()
                #os.chdir(os.path.join(os.getcwd(), 'MiDaS'))
            except ImportError:
                self.config = type('Config', (), {
                    'Horizontal_THRESH': 0.1134,
                    'Vertical_THRESH': 0.0772,
                    'DEPTH_THRESH': 50
                })()
                #os.chdir(os.path.join(os.getcwd(), 'MiDaS'))
        else:
            self.config = config

        self.width = 0
        self.long = 0

    def get_object_depth(self, depth_map, bbox):
        """计算物体 bbox 区域内的深度直方图众数"""
        x1, y1, x2, y2 = map(int, bbox)
        H, W = depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return 0

        obj_depth_crop = depth_map[y1:y2, x1:x2]
        valid_depths = obj_depth_crop[(obj_depth_crop > 0) & np.isfinite(obj_depth_crop)]
        if len(valid_depths) == 0:
            return 0

        try:
            num_bins = min(50, max(10, len(valid_depths) // 20))
            counts, bin_edges = np.histogram(valid_depths, bins=num_bins)
            max_bin_idx = np.argmax(counts)
            representative_depth = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2.0
            return representative_depth
        except Exception:
            return np.median(valid_depths)

    def compute_spatial_relations(self, objects, depth_map, image_size):
        """
    计算相对位置 (Pos_r) 和 绝对位置 (Pos_a)
    对应论文 Omni-Q: Eq (6) Absolute, Eq (7) Relative
    修改：先判断深度关系，再判断平面关系
         """
    
        H, W = image_size
        relations = []
        
        # --- 1. 计算每个物体的深度值 ---
        depth_vals = []
        for obj in objects:
            obj['depth_val'] = self.get_object_depth(depth_map, obj['bbox'])
            depth_vals.append(obj['depth_val'])
        
        # 计算深度范围
        depth_min, depth_max = np.min(depth_vals), np.max(depth_vals)
        DEPTH_RANGE = depth_max - depth_min if depth_max > depth_min else 1.0
        
        # --- 2. 绝对位置计算 (Absolute Position - Pos_a) ---
        # 绝对位置应包含深度信息
        for obj in objects:
            cx, cy = obj['center']
            
            # 2.1 计算深度位置
            depth_ratio = (obj['depth_val'] - depth_min) / DEPTH_RANGE if DEPTH_RANGE > 0 else 0.5
            
            if depth_ratio < 0.5:
                depth_pos = "the back"       # 深度比率小 -> 靠后
            elif depth_ratio < 0.67:
                depth_pos = "middle"         # 中间
            else:
                depth_pos = "the front"      # 深度比率大 -> 靠前
            
            obj['depth_pos'] = depth_pos
            obj['ratio'] = depth_ratio
            #print(depth_pos)
            
            # 2.2 计算平面位置
            others = [o for o in objects if o != obj]
            if not others:
                plane_pos = "center"
            else:
                leftmost = all(cx <= o['center'][0] for o in others)
                rightmost = all(cx >= o['center'][0] for o in others)
                
                if leftmost:
                    plane_pos = "leftmost"
                elif rightmost:
                    plane_pos = "rightmost"
                else:
                    # 九宫格/简单区域划分
                    h_str = "left" if cx < W/3 else ("right" if cx > 2*W/3 else "middle")
                    v_str = "top" if cy < H/3 else ("bottom" if cy > 2*H/3 else "center")
                    
                    if h_str == "middle" and v_str == "center":
                        plane_pos = "center"
                    elif h_str == "middle":
                        plane_pos = v_str
                    elif v_str == "center":
                        plane_pos = h_str
                    else:
                        plane_pos = f"{h_str} {v_str}"
            
            # 2.3 组合绝对位置描述
            obj['pos_a'] = f"{plane_pos} in {depth_pos}"
        
        # --- 3. 相对位置计算 (Relative Position - Pos_r) ---
        HORIZONTAL_RANGE = W
        VERTICAL_RANGE = H
        
        analy = []  # 用于调试，存储归一化的差异值

        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j: 
                    continue
                    
                obj_i = objects[i]
                obj_j = objects[j]
                rel = None
                diff_depth = None

                # 计算三个方向的差异
                if obj_i['depth_pos'] != obj_j['depth_pos']:
                    diff_depth = obj_i['depth_val'] - obj_j['depth_val']  # 正值表示i更近
                    
                dx = obj_i['center'][0] - obj_j['center'][0]          # 正值表示i在j右侧
                dy = obj_i['center'][1] - obj_j['center'][1]          # 正值表示i在j下方

               
                # 归一化差异值
                #norm_depth = abs(diff_depth) / DEPTH_RANGE if DEPTH_RANGE > 0 else 0
                norm_dx = abs(dx) / HORIZONTAL_RANGE if HORIZONTAL_RANGE > 0 else 0
                norm_dy = abs(dy) / VERTICAL_RANGE if VERTICAL_RANGE > 0 else 0
                
                #analy.append([norm_depth, norm_dx, norm_dy])
                
                # 根据论文描述：先判断深度关系，再判断平面关系
                # 1. 首先判断深度关系
                if diff_depth and abs(diff_depth)> self.config.DEPTH_THRESH:
                    # 深度差异显著，优先判断前后关系
                    if diff_depth > 0:
                        rel = "in front of"  # i比j近
                    else:
                        rel = "behind"       # i比j远
                        
                # 2. 深度差异不显著，判断平面关系
                elif norm_dx > self.config.Horizontal_THRESH and norm_dx > norm_dy:
                    # 水平差异显著
                    if dx > 0:
                        rel = "to the right of"
                    else:
                        rel = "to the left of"
                        
                elif norm_dy > self.config.Vertical_THRESH:
                    # 垂直差异显著
                    if dy > 0:
                        rel = "below"
                    else:
                        rel = "above"
                        
                else:
                    rel = "near"  # 没有显著差异
                
                # 调试输出
                #print('$'*30)
                #print(f"Objects: {obj_i.get('raw_desc', 'obj_'+str(i))} vs {obj_j.get('raw_desc', 'obj_'+str(j))}")
                #print(f"Normalized differences - depth: {norm_depth:.4f}, horizontal: {norm_dx:.4f}, vertical: {norm_dy:.4f}")
                #print(f"Thresholds - depth: {self.config.DEPTH_THRESH}, horizontal: {self.config.Horizontal_THRESH}, vertical: {self.config.Vertical_THRESH}")
                #print(f"Relation: {rel}")
                #print('@'*30)
                
                relations.append((obj_i['id'], obj_j['id'], rel))
                '''
                print('87@78'*15)
                #if diff_depth:
                print(diff_depth)
                print(f"{obj_i['depth_pos']}---{obj_j['depth_pos']}")
                print(f"{obj_i['ratio']}---{obj_j['ratio']}")
                print(f"{obj_i['depth_val']}---{obj_j['depth_val']}")
                print(f"{obj_i['raw_desc']}==={rel}===={obj_j['raw_desc']}")
                print(f"{len(obj_i['expressions'])}==={rel}===={len(obj_j['expressions'])}")
                print('90@09'*15)
                '''
        return relations#, analy

    def estimate_depth(self, image_path):
        # 1. 读取原图（RGB, [0,1], numpy）
        original_image_rgb = utils.read_image(image_path)

        # 2. 用 transform 预处理（得到 numpy 数组，形状 (3, net_h, net_w)）
        image = self.transform({"image": original_image_rgb})["image"]

        # 3. 转成 tensor，加 batch 维，移到 device（与 process() 内部一致）
        sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

        # 4. 推理
        with torch.no_grad():
            prediction = self.model.forward(sample)

        # 5. 插值回原图尺寸（注意 target_size 是 (W,H)，interpolate 的 size 要 (H,W)）
        target_size = original_image_rgb.shape[1::-1]  # (W, H)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],          # (H, W)
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return prediction
#print(os.getcwd())
#sr = SpatialRelationModule3D()
#print(os.getcwd())
#a = sr.estimate_depth('../data_set/VG datasets/mscoco2014/images/unc_train/COCO_train2014_000000002860.jpg')

