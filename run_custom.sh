EXP_NAME="baseline"
SCENE_NAME="sliding_cube_motion_range_0.0_baseline_scale_2.0_rotation_angle_0.0"
METHOD_NAME="Static"
RUN_NAME="${SCENE_NAME}_${METHOD_NAME}"
python runner.py \
    --config_file configs/custom/${METHOD_NAME}/vanilla1.yaml \
    --group ${EXP_NAME} \
    --name ${RUN_NAME} \
    --dataset data/custom/${SCENE_NAME} \
    --output_dir output/custom/${EXP_NAME}/${SCENE_NAME}/${METHOD_NAME}

# --slurm_script slurms/custom.sh \