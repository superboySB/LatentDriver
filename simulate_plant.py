import hydra
from omegaconf import OmegaConf

from src.utils.utils import update_waymax_config
from src.utils.mlp import MLP
from src.simulator.waymo_env import WaymoEnv
from src.utils.viz import plot_image


import matplotlib.pyplot as plt
from tqdm import tqdm
import mediapy
from torch import nn
import pytorch_lightning as pl
import torch
import time
import numpy as np
import os
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModel,
)

class BertEncoder(nn.Module):
    
    def __init__(self,
                 type_name='bert-mini',
                 online=False,
                 attributes = 6,
                 embedding_type = 'bicycle',
                 **kwargs,):
        super(BertEncoder, self).__init__()
        # debug: split bicycle or wpts by embedding_type, 5 if bicycle, 4 is wpts
        # we think bicycle needs sdc_mask but not padding, but wpts needs padding not sdc_mask
        # 2.1 changed for bicycle
        self.control_type = embedding_type


        # self.control_type = 'bicycle'
        embedding_type = 5
        # end debug
        # debug add sdc_mask
        self.embedding_type = embedding_type # routes, vehicles road and sdc, dont need to add paddings

        print("online!!!")
        config_bert = AutoConfig.from_pretrained(os.path.join('prajjwal1',type_name))  # load config from hugging face model
        self.model = AutoModel.from_config(config=config_bert)   
            
            
        self.hidden_size =  config_bert.hidden_size 
        n_embd = config_bert.hidden_size        
        self.cls_emb = nn.Parameter(
            torch.randn(1, attributes+1)
        )# +1 beacuse of the type
        # token embedding
        self.tok_emb = nn.Linear(attributes, n_embd)
        # object type embedding
        self.obj_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, attributes))
                for _ in range(self.embedding_type)
            ]
        )
        self.obj_emb = nn.ModuleList(
            [nn.Linear(attributes, n_embd) for _ in range(self.embedding_type)]
        )
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x, return_full_length=False):
        # if x.dim() == 4:
        #     x = x.reshape(-1,x.shape[-2],x.shape[-1])
        # obs.shape [bs,max_len_routes+max_num_vehicles, 1+6]
        # typeis for 1:routes, 2:vehicles 3:road_graph 4:trajs
        B,_,_ = x.shape 
        x = torch.cat([self.cls_emb.repeat(B,1,1), x],dim=1)
        input_batch_type = x[:, :, 0]  # car or map
        input_batch_data = x[:, :, 1:]
        car_mask = (input_batch_type == 2).unsqueeze(-1)
        road_graph_mask = (input_batch_type == 3).unsqueeze(-1)
        route_mask = (input_batch_type == 1).unsqueeze(-1)
        sdc_mask = (input_batch_type == 4).unsqueeze(-1)
        padding_mask = (input_batch_type == 0).unsqueeze(-1)
        # get other mask
        other_mask = torch.logical_not(torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_or(route_mask, car_mask), road_graph_mask), sdc_mask), padding_mask))
        # other_mask = torch.logical_and(route_mask.logical_not(), car_mask.logical_not(), road_graph_mask.logical_not(),sdc_mask.logical_not(),padding_mask.logical_not())
        if self.control_type == 'bicycle':
            masks = [car_mask, route_mask, road_graph_mask,other_mask,sdc_mask]
        elif self.control_type == 'waypoint':
            masks = [car_mask, route_mask, road_graph_mask,other_mask,padding_mask]

        # get size of input
        (B, O, A) = (input_batch_data.shape)  # batch size, number of objects, number of attributes

        # embed tokens object wise (one object -> one token embedding)
        input_batch_data = rearrange(
            input_batch_data, "b objects attributes -> (b objects) attributes"
        )
        embedding = self.tok_emb(input_batch_data)
        embedding = rearrange(embedding, "(b o) features -> b o features", b=B, o=O)

        # create object type embedding
        obj_embeddings = [
            self.obj_emb[i](self.obj_token[i]) for i in range(self.embedding_type)
        ]  # list of a tensors of size 1 x features

        # add object type embedding to embedding (mask needed to only add to the correct tokens)
        embedding = [
            (embedding + obj_embeddings[i]) * masks[i] for i in range(self.embedding_type)
        ]
        # debug dropout sdc_obj_emb
        # embedding[-1] = self.drop(embedding[-1])
        embedding = torch.sum(torch.stack(embedding, dim=1), dim=1)

        # embedding dropout
        embedding = self.drop(embedding)
        # Transformer Encoder; use embedding for hugging face model and get output states and attention map
        output = self.model(**{"inputs_embeds": embedding}, output_attentions=True)
        x, attn_map = output.last_hidden_state, output.attentions
        # fea = x[sdc_mask.squeeze(-1)]
        if not return_full_length:
            fea = x[:, 0, :]
            return fea
        elif return_full_length:
            return x


class Simple_driver(pl.LightningModule):
    def __init__(self,
                 action_space,
                 hidden_channels:list[int] = [64,64],
                 control_type:str = None,
                 optimizer = None,
                 encoder = None,
                 scheduler = None,
                 **kwarg,
                 ):
        super().__init__()
        # 兼容 action_space 为 dict 或对象
        if isinstance(action_space, dict):
            control_type_val = action_space.get('dynamic_type', None)
        else:
            control_type_val = getattr(action_space, 'dynamic_type', None)
        # 优先使用传入的 control_type 参数，否则用 action_space.dynamic_type
        self.control_type = control_type if control_type is not None else control_type_val
        assert self.control_type == 'waypoint'
        out_dim = 3
        print(f"[Simple_driver] control_type: {self.control_type}, out_dim: {out_dim}")
        
        self.bert = BertEncoder(**encoder)
        self.optim_conf = optimizer
        self.sched_conf = scheduler
        self.lr = kwarg['learning_rate']
        
        self.fc_head =MLP( 
            in_channels=self.bert.hidden_size,
            hidden_channels=hidden_channels[0],
            out_channels=out_dim,
            layer_num=len(hidden_channels),
            activation=nn.ReLU(),
            norm_type='BN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=True,) 
            
        self.out_dim = out_dim

    def forward(
        self,
        states, # bs , seq_len, state_attributes, state_dim
    ):
        batch_size, seq_length, state_elements, state_dims = states.shape[0], states.shape[1], states.shape[2], states.shape[3]
        x = states.reshape(batch_size*seq_length,state_elements,state_dims)
        fea = self.bert(x)

        out = self.fc_head(fea)

        out = out.reshape(batch_size,seq_length,self.out_dim)
        return out

    def get_predictions(self, states, actions, timesteps, num_envs=1, **kwargs):
        state = states[:,-1:]
        out = self.forward(state)
        return out[:,-1]


class LTDSimulator:
    def __init__(self, model, config, batch_dims):
        self.env = WaymoEnv(
            waymax_conf=config.waymax_conf,
            env_conf=config.env_conf,
            batch_dims=batch_dims,
            ego_control_setting=config.ego_control_setting,
            metric_conf=config.metric_conf,
            data_conf=config.data_conf,
        )
        self.batch_dims = batch_dims
        self.size = batch_dims[0] * batch_dims[1]
        self.model = model
        self.model.eval()
        # put the policy model on the last available gpu
        if torch.cuda.is_available():
            self.device = 'cuda:{}'.format(torch.cuda.device_count()-1)
        self.model.to(self.device) 
        self.cfg = config
    
    def run(self):
        self.idx = 0
        while True:
            try:
                obs, obs_dict = self.env.reset()
                obs = obs.reshape(self.env.num_envs,-1,7)
                obs_depth,obs_dim = obs.shape[1],obs.shape[-1]
                states = (obs).reshape(self.env.num_envs,-1,obs_depth,obs_dim)
                timesteps = torch.tensor([0] * self.env.num_envs, device=self.device, dtype=torch.long).reshape(
                    self.env.num_envs, -1
                )

                # dx, dy, dyaw
                assert self.cfg.action_space.dynamic_type == 'waypoint'
                actions = np.zeros((self.env.num_envs, 1,3))
                rewards = np.zeros((self.env.num_envs, 1,1))
                done_ = False
                self.T = 1
                a = time.time()
                

                while not done_:
                    rewards = np.concatenate(
                        [
                            rewards,
                            np.zeros((self.env.num_envs, 1)).reshape(self.env.num_envs, -1, 1),
                        ],
                        axis=1,
                    )        

                    # planT模型输入输出维度详解：
                    # 
                    # 输入维度 (188, 7) 的含义：
                    # - 188 = 20(route_segments) + 128(max_num_objects) + 40(max_roadgraph_segments)
                    #   - 20: 路径段数量 (max_route_segments)，表示ego车辆可以行驶的路径点
                    #   - 128: 最大车辆数量 (max_num_objects)，包括ego车辆和其他车辆
                    #   - 40: 最大道路图段数量 (max_roadgraph_segments)，表示道路网络信息
                    # 
                    # ========== 188个7维向量的具体顺序和组织方式 ==========
                    # 
                    # 索引范围: [0, 187] 总共188个向量
                    # 
                    # 1. 路径段 (索引 0-19，共20个): 
                    #    - 按时间顺序排列，从ego车辆当前位置开始，沿着预定义路径向前
                    #    - 基于ego车辆的log_trajectory生成，使用RDP算法降采样
                    #    - 每个段包含: [type_id=1, x, y, width, length, yaw, id]
                    #    - 距离ego车辆超过50米的段会被过滤掉
                    #    - type_id = 1 (route类型)
                    # 
                    # 2. 车辆 (索引 20-147，共128个): 
                    #    - 按原始数据中的顺序排列，但ego车辆会被特殊标记
                    #    - 通过sdc_idx = top_k(is_sdc, k=1)找到ego车辆在128个车辆中的索引
                    #    - ego车辆的类型标记为4，其他车辆标记为2
                    #    - 车辆特征包括: [type_id, x, y, width, length, yaw, speed]
                    #    - 超出视野范围(80m×20m)的车辆会被置零
                    #    - type_id = 2 (其他车辆) 或 4 (ego车辆)
                    # 
                    # 3. 道路图段 (索引 148-187，共40个): 
                    #    - 按距离ego车辆的距离排序，最近的优先
                    #    - 使用filter_topk_roadgraph_points函数选择最近的40个道路点
                    #    - 基于欧几里得距离计算，只考虑valid=True的点
                    #    - 道路图特征包括: [type_id=3, x, y, width, length, yaw, id]
                    #    - type_id = 3 (roadgraph类型)
                    # 
                    # ========== 7维特征向量的详细说明 ==========
                    # 
                    # 每个7维向量包含以下特征：
                    # - 第1维: 对象类型 (0:padding, 1:route, 2:vehicle, 3:roadgraph, 4:ego)
                    # - 第2-3维: x,y坐标 (相对于ego车辆的位置，已转换到ego坐标系)
                    # - 第4-5维: width,length (车辆尺寸，道路宽度等)
                    # - 第6维: yaw (朝向角度，度为单位，已转换到ego坐标系)
                    # - 第7维: speed (速度，仅车辆有此特征) 或 id (路径段/道路图的标识)
                    # 
                    # ========== 7维向量的预处理和变换详解 ==========
                    # 
                    # 1. 坐标系变换 (src/simulator/observation.py):
                    #    - 所有对象的坐标都转换到ego车辆为中心的坐标系
                    #    - 使用ObjectPose2D进行刚性变换
                    #    - 变换矩阵: pose_global2ego = combine_two_object_pose_2d(src_pose, dst_pose)
                    #    - 坐标变换: transformed_xy = transform_points(pts, pose_matrix)
                    #    - 角度变换: transformed_yaw = transform_yaw(-sdc_yaw, original_yaw)
                    # 
                    # 2. 角度单位转换:
                    #    - 原始yaw: 弧度制 (radian)
                    #    - 转换后yaw: 度制 (degree)
                    #    - 转换公式: yaw_degree = yaw_radian * 180 / π
                    #    - 代码实现: yaw = sdc_obs.trajectory.yaw * 180 / np.pi
                    # 
                    # 3. 视野范围过滤:
                    #    - 视野范围: 80m × 20m (可配置)
                    #    - 过滤条件: |x| > 40m 或 |y| > 10m
                    #    - 超出范围的对象被置零 (padding)
                    #    - 代码实现: padding_exceed() 函数
                    # 
                    # 4. 数值处理:
                    #    - 无数值归一化: 所有数值保持原始尺度
                    #    - 无标准化: 没有进行z-score或min-max归一化
                    #    - 直接使用原始物理单位: 米、度、米/秒
                    # 
                    # 5. 具体变换过程:
                    #    a) 车辆数据 (get_vehicle_obs):
                    #       - xy: 已转换到ego坐标系
                    #       - yaw: 弧度→度，已转换到ego坐标系
                    #       - width/length: 保持原始值
                    #       - speed: 保持原始值
                    #    
                    #    b) 道路图数据 (downsampled_elements_transformation):
                    #       - xy: 全局坐标→ego坐标系
                    #       - yaw: 弧度→度，全局→ego坐标系
                    #       - width/length: 保持原始值
                    #       - id: 保持原始值
                    #    
                    #    c) 路径段数据:
                    #       - 与道路图数据相同的变换过程
                    # 
                    # ========== 重要说明 ==========
                    # 
                    # - 坐标系变换是必须的，确保模型以ego车辆为参考点
                    # - 角度单位转换是必要的，因为模型训练时使用度制
                    # - 视野范围过滤提高计算效率，减少无关信息
                    # - 无数值归一化保持物理意义的直观性
                    # - 所有变换都在数据预处理阶段完成，模型直接使用变换后的数据
                    # 
                    # ========== 实际数据举例说明 ==========
                    # 
                    # 以下是一个实际场景中188个向量的第一维(类型ID)分布示例：
                    # [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    #  2.0, 2.0, 0.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    #  ... (大量0.0 padding) ..., 
                    #  3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    #  3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
                    # 
                    # 分析说明：
                    # - 索引 0-19 (路径段区域): 2个1.0(route) + 18个0.0(padding) = 20个
                    # - 索引 20-147 (车辆区域): 14个有效车辆(2.0/4.0) + 114个0.0(padding) = 128个
                    #   * 其中4.0表示ego车辆，2.0表示其他车辆
                    # - 索引 148-187 (道路图段区域): 16个3.0(roadgraph) + 24个0.0(padding) = 40个
                    # 
                    # ========== 重要说明 ==========
                    # 
                    # 这个顺序是固定的，不能随意打乱，因为：
                    # 1. BertEncoder中的掩码处理依赖于对象在序列中的位置
                    # 2. 不同类型的对象使用不同的嵌入层
                    # 3. 模型最终输出取第一个token (CLS token，位于序列开头)
                    # 4. 路径段、车辆、道路图的相对位置关系对模型理解场景很重要
                    # 5. CLS token有独立的类型维度，不参与对象类型掩码处理
                    # 
                    # ========== 模型输出和动作范围限制机制详解 ==========
                    # 
                    # 输出维度 (3,) 的含义 - waypoint模式的动作：
                    # - dx: x方向位移 (米)，范围[-0.14, 6.0]，在ego车辆坐标系中
                    # - dy: y方向位移 (米)，范围[-0.35, 0.35]，在ego车辆坐标系中
                    # - dyaw: 朝向角变化 (弧度)，范围[-0.15, 0.15]
                    # 
                    # ========== 模型输出处理流程 ==========
                    # 
                    # 1. 模型原始输出：
                    #    - planT模型直接输出3维向量 [dx, dy, dyaw]
                    #    - 输出值范围：理论上可以是任意实数 (无限制)
                    #    - 输出格式：torch.Tensor，形状为 (batch_size, 3)
                    # 
                    # 2. 输出后处理：
                    #    - 转换为numpy数组：action.detach().cpu().numpy()
                    #    - 直接传递给环境：control_action = action
                    #    - 无额外的normalization或denormalization步骤
                    # 
                    # 3. 环境中的动作处理：
                    #    - 环境接收原始动作值
                    #    - 通过DeltaLocal动力学模型处理动作
                    #    - 在动力学模型中实现范围限制
                    # 
                    # ========== 动作范围限制机制 ==========
                    # 
                    # 范围限制在以下位置实现：
                    # 
                    # 1. 配置文件定义 (simulate_plant.yaml):
                    #    action_ranges: [[-0.14, 6], [-0.35, 0.35], [-0.15,0.15]]
                    # 
                    # 2. 动力学模型裁剪 (waymax/dynamics/delta.py):
                    #    def _clip_values(self, action: jax.Array) -> jax.Array:
                    #        x = jnp.clip(action[..., 0], min=-0.14, max=6.0)
                    #        y = jnp.clip(action[..., 1], min=-0.35, max=0.35)
                    #        yaw = geometry.wrap_yaws(action[..., 2])  # 角度包装
                    #        return jnp.stack([x, y, yaw], axis=-1)
                    # 
                    # 3. 动作空间规范 (src/simulator/waymo_base.py):
                    #    self.action_space_ = gym.spaces.Box(
                    #        low=np.array([-0.14, -0.35, -0.15]),
                    #        high=np.array([6.0, 0.35, 0.15]),
                    #        shape=(3,), dtype=np.float32
                    #    )
                    # 
                    # ========== 动作作用机制 ==========
                    # 
                    # 1. 通过DeltaLocal动力学模型将waypoint动作转换为全局坐标
                    #    - 使用当前ego车辆的yaw角度构建旋转矩阵
                    #    - 将局部坐标的dx,dy转换为全局坐标
                    # 2. 更新ego车辆的位置: new_x = x + dx, new_y = y + dy
                    # 3. 更新ego车辆的朝向: new_yaw = yaw + dyaw
                    # 4. 计算新的速度: vel_x = dx/dt, vel_y = dy/dt (dt=0.1秒)
                    # 5. 这些动作直接控制ego车辆在下一个时间步的位置和朝向
                    # 6. 动作会被限制在配置的范围内，超出范围会被裁剪
                    # 
                    # ========== 重要说明 ==========
                    # 
                    # - 模型输出是原始值，不是normalized值
                    # - 范围限制在动力学模型层面实现，不在模型输出层面
                    # - 如果模型输出超出范围，会被自动裁剪到有效范围内
                    # - 这种设计允许模型学习更自然的动作分布，同时保证安全性
                    with torch.no_grad():
                        action = self.model.get_predictions(
                            torch.tensor(states,device =self.device),
                            torch.tensor(actions,device =self.device),
                            timesteps.to(dtype=torch.long),
                            num_envs=self.env.num_envs
                        )
                    
                        
                    if isinstance(action,torch.Tensor):
                        action = action.detach().cpu().numpy()
                    control_action = action

                    obs, obs_dict,rew, done, info = self.env.step(control_action,show_global=True)
                    actions = np.concatenate([actions,action[:,np.newaxis,...]],axis=1)
                    # actions[:, -1] = action
                    obs = obs.reshape(self.env.num_envs,-1,7)
                    state = (obs.reshape(self.env.num_envs,-1,obs_depth,obs_dim))
                    states = np.concatenate([states,state],axis=1)
                    
                    timesteps = torch.cat(
                        [
                            timesteps,
                            torch.ones((self.env.num_envs, 1), device=self.device, dtype=torch.long).reshape(
                                self.env.num_envs, 1
                            )
                            * (self.T),
                        ],
                        dim=1,
                        )

                    self.T+=1
                    done_ =done[-1]
                self.idx += 1
                print('Processed: ', self.idx, 'th batch, Time: ', time.time()-a, 's')
                
                self.render("video", self.cfg.method.model_name)
                    
            except StopIteration:
                print("StopIteration")
                break

    def render(self, vis, model_name):
        def save_video():
            # for full video
            imgs = []
            margin = 30
            vis_config = dict(
                front_x=margin,
                back_x=margin,
                front_y=margin,
                back_y=margin,
                px_per_meter=20,
                show_agent_id=True,
                center_agent_idx=-1,
                verbose=False
            )
            for state in tqdm(self.env.states):
                imgs.append(plot_image(state = state,batch_idx=j,viz_config=vis_config))
                mediapy.write_video(name+'.mp4',imgs , fps=10)
            print('Saved Video: ', name)
            
        if vis==False:
            return
        assert vis in ['image','video'], "vis must be either 'image' or 'video'"
        root_folder = f'vis_results/{model_name}/{vis}/'
        os.makedirs(root_folder, exist_ok=True)
        os.makedirs(root_folder+'straight_', exist_ok=True)
        os.makedirs(root_folder+'turning_left', exist_ok=True)
        os.makedirs(root_folder+'turning_right', exist_ok=True)
        os.makedirs(root_folder+'U-turn_left', exist_ok=True)
        for j in range(self.size):
            intention = self.env.intention_label[j]
            if intention == 'straight_':
                # sub_folder = root_folder+'straight_'
                sub_folder = None
            elif intention == 'turning_left':
                sub_folder = root_folder+'turning_left'
            elif intention == 'turning_right':
                sub_folder = root_folder+'turning_right'
            elif intention == 'U-turn_left':
                sub_folder = root_folder+'U-turn_left'
            else: sub_folder = None
            if sub_folder is not None:
                scen_id = self.env.get_env_idx(j)
                is_offroad = self.env.metric.info_hack['metric/offroad_rate'].reshape(-1)[j]
                is_collision = self.env.metric.info_hack['metric/collision_rate'].reshape(-1)[j]
                is_ar90 = self.env.metric.info_hack['metric/arrival_rate90'].reshape(-1)[j]
                name = f'{sub_folder}/AR{is_ar90}_OR{is_offroad}_CR{is_collision}_{scen_id}'
                save_video()
                

@hydra.main(version_base=None, config_path=".", config_name="simulate_plant")
def simulate(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    print(cfg)
    
    model = Simple_driver(**cfg.method)
    cfg = update_waymax_config(cfg)
    if cfg.ckpt_path is not None:
        model.load_state_dict(torch.load(cfg.ckpt_path,map_location='cuda:{}'.format(torch.cuda.device_count()-1))['state_dict'])
        print(f'Loaded {cfg.ckpt_path}')
    else:
        print('No ckpt provided')
    
    runner = LTDSimulator(
        model = model,
        config=cfg,
        batch_dims=cfg.batch_dims,
    )

    runner.run()

if __name__ == '__main__':
    simulate()