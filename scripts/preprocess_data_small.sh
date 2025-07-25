#!/bin/bash

# =============================
# preprocess_small_validation_data.sh
#
# 用途：
#   只对 /workspace/tiny_waymo 目录下的3个tfrecord文件进行预处理，
#   输出到 /workspace/tiny_waymo_processed 和 /workspace/tiny_waymo_intention_label。
#   适用于小规模验证/测试，不影响训练主流程。
#
# 使用方法：
#   先 source scripts/set_env_small.sh
#   再 bash scripts/preprocess_small_validation_data.sh
#
# 预处理完成后，可用如下命令进行测试（非训练）：
#   python simulate.py method=planT \
#       ++waymax_conf.path="${WOMD_VAL_PATH}" \
#       ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
#       ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
#       ++batch_dims=[7,125] \
#       ++ego_control_setting.npc_policy_type=idm \
#       ++method.ckpt_path='checkpoints/planT.ckpt'
# =============================

# 检查并删除已存在的输出目录，避免冲突
if [ -d "$PRE_PROCESS_VAL_PATH" ]; then
    rm -rf "$PRE_PROCESS_VAL_PATH"
fi
if [ -d "$INTENTION_VAL_PATH" ]; then
    rm -rf "$INTENTION_VAL_PATH"
fi

# 运行预处理脚本
PYTHONPATH=./ python src/preprocess/preprocess_data.py \
    ++batch_dims=[1,1] \
    ++waymax_conf.path=${WOMD_VAL_PATH} \
    ++waymax_conf.drop_remainder=False \
    ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
    ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}"

# =============================
# 说明：
# - 运行本脚本前，请确保已 source scripts/set_env_small.sh。
# - 输出目录会自动创建（如已存在需手动删除以避免冲突）。
# - 预处理完成后，直接用上述simulate.py命令即可进行测试。
# - batch_dims=[1,3] 适合3个文件的小批量处理，simulate.py时可根据实际需求调整batch_dims。
# ============================= 