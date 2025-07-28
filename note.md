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
        ++method.ckpt_path='checkpoints/planT.ckpt' 
```
最后加`vis="video"`可以打印mp4。根据性能有两个关键参数：
- batch_dims[0] 必须 ≤ 物理设备数（jax.local_device_count()）。
- batch_dims[1] 是每个设备上并行的环境/样本数，直接影响单卡显存消耗。

# 用于重构
维护一个测试planT的最小版本
```sh
python simulate_plant.py\
        ++waymax_conf.path="/workspace/tiny_waymo/tf_example/uncompressed_tf_example_validation_validation_tfexample.tfrecord-00000-of-00150"\
        ++data_conf.path_to_processed_map_route="/workspace/tiny_waymo/processed" \
        ++metric_conf.intention_label_path="/workspace/tiny_waymo/intention_label" \
        ++batch_dims=[1,16] \
        ++method.ckpt_path="checkpoints/planT.ckpt"
```