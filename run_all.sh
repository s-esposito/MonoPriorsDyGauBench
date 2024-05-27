# sbatch slurms/dnerf_trex_Curve_vanilla1.sh
# sbatch slurms/dnerf_trex_HexPlane_vanilla1.sh
# sbatch slurms/dnerf_trex_FourDim_vanilla1.sh
# sbatch slurms/dnerf_trex_MLP_vanilla1.sh
# sbatch slurms/dnerf_trex_TRBF_nodecoder1.sh
# sbatch slurms/dnerf_trex_TRBF_vanilla1.sh

# sbatch slurms/dnerf_trex_Curve_static_cube_moving_camera.sh
# sbatch slurms/dnerf_trex_FourDim_static_cube_moving_camera.sh
# sbatch slurms/dnerf_trex_HexPlane_static_cube_moving_camera.sh
# sbatch slurms/dnerf_trex_MLP_vanilla1_static_cube_moving_camera.sh
# sbatch slurms/dnerf_trex_TRBF_nodecoder1_static_cube_moving_camera.sh
# sbatch slurms/dnerf_trex_TRBF_static_cube_moving_camera.sh

# Make list of methods

declare -a methods=( \
    "Curve" \
    "FourDim" \
    "HexPlane" \
    "MLP" \
    "TRBF_nodecoder" \
    #"TRBF" \
)

declare -a scenes=( \
    #"static_cube_moving_camera_textured" \
    #"static_cube_moving_camera_textured_brick" \
    #"dynamic_cube_moving_camera_textured" \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_0.0' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_0.5' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_1.0' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_2.0' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_5.0' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_10.0' \
    # 'dynamic_cube_dynamic_camera_textured_motion_range_20.0' \
    #"dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_0.0" \
    #"dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_2.0" \
    #"dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_5.0" \
    #"dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_10.0" \
    #"dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_20.0" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_0.0_baseline_scale_0.1" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_0.0_baseline_scale_0.5" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_0.0_baseline_scale_2.0" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_5.0_baseline_scale_0.1" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_5.0_baseline_scale_0.5" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_5.0_baseline_scale_2.0" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_10.0_baseline_scale_0.1" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_10.0_baseline_scale_0.5" \
    "dynamic_cube_dynamic_camera_textured_motion_range_nvs_no_spiral_radius_1.0_motion_range_10.0_baseline_scale_2.0" \
)

for scene in "${scenes[@]}"
do
    for method in "${methods[@]}"
    do
        variant="dnerf/custom/${method}/${scene}"
        exp_group_name="vanilla"
        exp_name="${scene}_${method}"
        python runner.py \
            --config_file configs/dnerf/custom/${method}/vanilla1.yaml \
            --group ${exp_group_name}_${scene} \
            --name ${exp_name} \
            --dataset data/${scene} \
            --slurm_script slurms/dnerf_custom.sh \
            --output_dir output/dnerf/${exp_group_name}/${scene}/${method}
    done
done