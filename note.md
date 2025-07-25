# 复现笔记
## 配置
基于SB-DriveSim的环境配一下试试
## 测试
```sh
python simulate.py method=planT \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[7,125] \
        ++ego_control_setting.npc_policy_type=idm \
        ++method.ckpt_path='checkpoints/planT.ckpt'
```