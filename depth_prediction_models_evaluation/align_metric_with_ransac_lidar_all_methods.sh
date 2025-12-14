scenes=(
    apple
    backpack
    block
    creeper
    handwavy
    haru-sit
    mochi-high-five
    paper-windmill
    pillow
    spin
    sriracha-tree
    teddy
)
# Caution align_metric_with_ransac_lidar.py need to be changed for depth_pro and moge because of different metric folder structure
methods=(
    depth_pro
    moge
    # mega_sam
    # mega_sam_depth_pro
    # mega_sam_itwild
    # metric_aligned_depth_anything_colmap_depth
    # metric_aligned_depth_anything_v2
    # unidepth2
    # video_depth_anything
    # unidepth2_aligned_relative_video_depth_anything
    # unidepth2_aligned_depth_anything2_colmap_focall
    # video_depth_anything_aligned_depth_pro
)

for method in ${methods[@]}; do
    echo "Processing method: $method"
    # Call different scripts based on the depth method
    if [[ "$method" == "depth_pro" || "$method" == "moge" ]]; then
        echo "Using depth_pro or moge"
        for seq in ${scenes[@]}; do
            python align_metric_with_ransac_lidar.py \
            /home/geiger/gwb215/datasets/iphone/$seq/depth/1x \
            /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$method/metric/1x \
            /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/ransac_lidar_aligned_$method/1x
        done
    else
        echo "Using N_O_T depth_pro or moge"
        for seq in ${scenes[@]}; do
            python align_metric_with_ransac_lidar.py \
            /home/geiger/gwb215/datasets/iphone/$seq/depth/1x \
            /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$method/1x \
            /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/ransac_lidar_aligned_$method/1x
        done
    fi
done