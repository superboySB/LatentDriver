# 复现笔记
## 配置
基于V-Max(https://github.com/superboySB/V-Max/blob/main/note.md)的环境配一下试试
```sh
docker run -itd --privileged --gpus all --net=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro --shm-size=4g \
  -v /home/dzp/Public/tiny_waymo:/workspace/tiny_waymo \
  --name dzp-waymax-0717 \
  dzp_waymax:0717 \
  /bin/bash

docker exec -it dzp-waymax-0717 /bin/bash

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

python setup.py install

python tools/check_tf_jax_torch.py
# Output
# tf on gpu is available? False
# jax on gpu is available? True
# torch on gpu is available? True

source scripts/set_env_small.sh

bash scripts/preprocess_data_small.sh

```

## 测试
```sh
source scripts/set_env_small.sh

python simulate.py method=planT \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[1,16] \
        ++ego_control_setting.npc_policy_type=idm \
        ++method.ckpt_path='checkpoints/planT.ckpt' \
        vis=video
```
在你的代码和 JAX 框架下：
- batch_dims[0] 必须 ≤ 物理设备数（jax.local_device_count()）。
- batch_dims[1] 是每个设备上并行的环境/样本数，直接影响单卡显存消耗。