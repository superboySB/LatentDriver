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