import networkx as nx
import random
from typing import List, Dict, Any
import random



class SpatialGraphModule:
    def __init__(self, config=None):
        
        if config is None:
            try:
                from config.config import OmniQConfig
                self.config = OmniQConfig()
            except ImportError:
                print("[Warning] No config provided, using default values")
                # 创建默认配置
                self.config = type('Config', (), {
                    'QUERIES_PER_IMAGE': 48,
                    'MAX_VISITED_NODES': 2
                })()
                #修改
        else:
            self.config = config
        print(f"[Module] Initializing Spatial Graph (Graph-Based Query Generation)...")
        self.graph = nx.DiGraph()

    def build_and_traverse(self, objects, relations):
        """
        构建图并生成查询
        输入:
            objects: 包含 'id', 'expressions'(列表), 'pos_a'(自然语言字符串) 的字典列表
            relations: (src_id, dst_id, relation_str) 的列表
        """
        self.graph.clear()
        
        # 1. 构建图 (Construct Graph)
        # -------------------------------------------------
        for obj in objects:
            obj_id = obj['id']
            if obj['expressions'] != []:
                content = obj['expressions']
            else :
                content = obj['category']
            # [顶点 A]: Exp 节点 (代表对象)
            # content 存储该对象所有的表达式列表
            self.graph.add_node(obj_id, 
                                type='exp', 
                                content=content,
                                raw_obj=obj)
            
            # [顶点 B]: Absolute Position 节点 (代表绝对位置)
            # 注意：这里的 pos_a 是上一模块输出的自然语言 (如 "top left corner")
            pos_a_text = obj.get('pos_a')
            if pos_a_text:
                abs_node_id = f"abs_{obj_id}"
                self.graph.add_node(abs_node_id, 
                                    type='absolute_position', 
                                    content=[pos_a_text])
                
                # [边]: Exp -> Absolute Position
                # 建立关联，表示 "obj located at pos"
                self.graph.add_edge(obj_id, abs_node_id, relation=None)
                self.graph.add_edge(abs_node_id, obj_id, relation=None)  
        # [边]: Relative Position (Exp -> Exp)
        for src, dst, rel in relations:
            if self.graph.has_node(src) and self.graph.has_node(dst):
                self.graph.add_edge(src, dst, relation=rel)

        # 2. 遍历生成查询 (Generate Queries)
        # -------------------------------------------------
        queries = []
        # 获取所有类型 的节点作为合法的遍历起点
        
        all_nodes_id = list(self.graph.nodes())
        
        if not all_nodes_id:
            return []

        # 循环多次以覆盖不同的组合
        EPOCHS = self.config.QUERIES_PER_IMAGE*10


        '''
        p-q 标签
        tmp_pseudo_train_sample = [image_file, 'useless placeholder', object['bbox'][:4],
                                                   description_string, 'useless placeholder']
                        all_candidate.append(tmp_pseudo_train_sample)
        ''' 
        final_label = []
        
        # 遍历图中每一个节点，进行广度优先的一跳组合
        #print(len(self.graph.nodes))
        for start_node_id in self.graph.nodes:
            current_data = self.graph.nodes[start_node_id]
            #print(f"Node {start_node_id} has {self.graph.out_degree(start_node_id)} edges")
           
            # 情况1：当前节点是 exp（描述对象）
            if current_data['type'] == 'exp':
                #print(f"the frist on {len(current_data['content'])}")
                if not current_data['content']:
                    continue

                # 记录该 exp 的定位信息（无论有多少个 content，都共用同一组定位信息）
                bbox = current_data['raw_obj']['bbox']
                img_id = current_data['raw_obj']['image_id']
                shape = current_data['raw_obj']['shape']

                # 遍历该 exp 节点的所有描述，生成单节点查询
                for curr_str in current_data['content']:
                    final_label.append([img_id, shape, bbox, curr_str, 'useless placeholder'])

                # 遍历所有邻居，生成两节点组合查询（一跳），并遍历所有描述组合
                for neighbor_id in self.graph.neighbors(start_node_id):
                    neighbor_data = self.graph.nodes[neighbor_id]
                    #print(f"the second on {len(neighbor_data['content'])}")
                    if not neighbor_data['content']:
                        continue

                    edge_data = self.graph.get_edge_data(start_node_id, neighbor_id)
                    rel_str = edge_data.get('relation', '')

                    # 邻居是 absolute_position：不需要关系，直接拼接
                    if neighbor_data['type'] == 'absolute_position':
                        # 笛卡尔积：exp 的每个描述 × 位置节点的每个描述
                        for curr_str in current_data['content']:
                            for next_str in neighbor_data['content']:
                                final_query = f"{curr_str} {next_str}"
                                final_label.append([img_id, shape, bbox, final_query, 'useless placeholder'])

                    # 邻居也是 exp：使用边关系连接
                    else:
                        for curr_str in current_data['content']:
                            for next_str in neighbor_data['content']:
                                final_query = f"{curr_str} {rel_str} {next_str}"
                                final_label.append([img_id, shape, bbox, final_query, 'useless placeholder'])

            # 情况2：当前节点是 absolute_position
            elif current_data['type'] == 'absolute_position':
                if not current_data['content']:
                    continue

                # 遍历该位置节点的所有描述
                for curr_pos_str in current_data['content']:
                    for neighbor_id in self.graph.neighbors(start_node_id):
                        neighbor_data = self.graph.nodes[neighbor_id]
                        # 只关心邻居中的 exp 节点
                        if neighbor_data['type'] != 'exp' or not neighbor_data['content']:
                            continue

                        # 使用邻居 exp 的定位信息
                        bbox = neighbor_data['raw_obj']['bbox']
                        img_id = neighbor_data['raw_obj']['image_id']
                        shape = neighbor_data['raw_obj']['shape']

                        # 位置描述 × exp 描述 的笛卡尔积
                        for next_exp_str in neighbor_data['content']:
                            final_query = f"{curr_pos_str} {next_exp_str}"
                            final_label.append([img_id, shape, bbox, final_query, 'useless placeholder'])

                #if final_query:
                    #queries.append(final_query)

        # 3. 去重与筛选
        #unique_queries = list(set(queries))
        #return unique_queries[:self.config.QUERIES_PER_IMAGE]
        
        #final_label = list (set(final_label))
        from transformers import BertTokenizer

        # 加载 BERT 分词器（根据语种选择，中文用 bert-base-chinese，英文用 bert-base-uncased）
        tokenizer = BertTokenizer.from_pretrained('./GRiT/models/bert-base-uncased')   

        mem = []
        sel = []
        contt = 0
        print(len(final_label),'or')
        for l in final_label:
            # 1. 根据 l[3] 去重（保留首次出现）
            if l[3] in mem:
                contt += 1
                continue
            
            mem.append(l[3])
            # 2. 用 BERT 分词器对文本字段分词，并检查 token 长度 < 20
            text = l[3]  # 假设文本在每行的第一个位置，根据实际修改
            tokens = tokenizer.tokenize(text)
            inputs = tokenizer(text, add_special_tokens=True)
            if len(inputs['input_ids']) >= 20:   # 实际输入长度超过或等于20则跳过  
                #print(text)
                #print(len(inputs['input_ids']))
                continue

            # 满足条件：记录并加入结果
            
            sel.append(l)

        final_label = sel
        return random.sample(final_label,min(len(final_label),self.config.QUERIES_PER_IMAGE))