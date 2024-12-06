declare -a methods=( \
    "Curve" \
    "FourDim" \
    "HexPlane" \
    "MLP" \
    "TRBF_nodecoder" \
    "TRBF" \
    "Static" \
)

declare -a scenes=( \
    "sliding_cube_motion_range_10.0_baseline_scale_0.1_rotation_angle_0.0" \
    "sliding_cube_motion_range_10.0_baseline_scale_0.1_rotation_angle_3.14" \
    "sliding_cube_motion_range_10.0_baseline_scale_1.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_10.0_baseline_scale_1.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_10.0_baseline_scale_2.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_10.0_baseline_scale_2.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_10.0_baseline_scale_0.5_rotation_angle_0.0" \
    "sliding_cube_motion_range_10.0_baseline_scale_0.5_rotation_angle_3.14" \
    "sliding_cube_motion_range_10.0_baseline_scale_0.3_rotation_angle_0.0" \
    "sliding_cube_motion_range_10.0_baseline_scale_0.3_rotation_angle_3.14" \

    "sliding_cube_motion_range_5.0_baseline_scale_0.1_rotation_angle_0.0" \
    "sliding_cube_motion_range_5.0_baseline_scale_0.1_rotation_angle_3.14" \
    "sliding_cube_motion_range_5.0_baseline_scale_1.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_5.0_baseline_scale_1.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_5.0_baseline_scale_2.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_5.0_baseline_scale_2.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_5.0_baseline_scale_0.5_rotation_angle_0.0" \
    "sliding_cube_motion_range_5.0_baseline_scale_0.5_rotation_angle_3.14" \
    "sliding_cube_motion_range_5.0_baseline_scale_0.3_rotation_angle_0.0" \
    "sliding_cube_motion_range_5.0_baseline_scale_0.3_rotation_angle_3.14" \

    "sliding_cube_motion_range_0.0_baseline_scale_0.1_rotation_angle_0.0" \
    "sliding_cube_motion_range_0.0_baseline_scale_0.1_rotation_angle_3.14" \
    "sliding_cube_motion_range_0.0_baseline_scale_1.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_0.0_baseline_scale_1.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_0.0_baseline_scale_2.0_rotation_angle_0.0" \
    "sliding_cube_motion_range_0.0_baseline_scale_2.0_rotation_angle_3.14" \
    "sliding_cube_motion_range_0.0_baseline_scale_0.5_rotation_angle_0.0" \
    "sliding_cube_motion_range_0.0_baseline_scale_0.5_rotation_angle_3.14" \
    "sliding_cube_motion_range_0.0_baseline_scale_0.3_rotation_angle_0.0" \
    "sliding_cube_motion_range_0.0_baseline_scale_0.3_rotation_angle_3.14" \
)

for batch in 1
do
    for scene in "${scenes[@]}"
    do
        for method in "${methods[@]}"
        do
            exp_group_name="custom_dataset_test_${batch}"
            exp_name="${scene}_${method}_batch_${batch}"
            python runner.py \
                --config_file configs/custom/${method}/vanilla1.yaml \
                --group ${exp_group_name}_${scene} \
                --name ${exp_name} \
                --dataset data/custom/${scene} \
                --slurm_script slurms/custom.sh \
                --output_dir output/custom/${exp_group_name}/${scene}/${method}
        done
    done
done