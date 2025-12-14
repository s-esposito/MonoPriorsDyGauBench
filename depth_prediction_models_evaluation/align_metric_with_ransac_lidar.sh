scenes=(
    # apple
    # backpack
    block
    # creeper
    # handwavy
    # haru-sit
    # mochi-high-five
    # paper-windmill
    # pillow
    # spin
    # sriracha-tree
    # teddy
)

# Choose one of the following depth methods:
# depth_pro, moge, mega_sam, mega_sam_depth_pro, mega_sam_itwild, metric_aligned_depth_anything_colmap_depth, 
# metric_aligned_depth_anything_v2, unidepth2, video_depth_anything, unidepth2_aligned_relative_video_depth_anything,
# unidepth2_aligned_depth_anything2_colmap_focall, video_depth_anything_aligned_depth_pro
DEPTH_METHOD=video_depth_anything_aligned_depth_pro

# Call different scripts based on the depth method
if [[ "$DEPTH_METHOD" == "depth_pro" || "$DEPTH_METHOD" == "moge" ]]; then
    echo "Using depth_pro or moge"
    for seq in ${scenes[@]}; do
        python align_metric_with_ransac_lidar.py \
        /home/geiger/gwb215/datasets/iphone/$seq/depth/1x \
        /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$DEPTH_METHOD/metric/1x \
        /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/ransac_lidar_aligned_$DEPTH_METHOD/1x
    done
else
    echo "Using N_O_T depth_pro or moge"
    for seq in ${scenes[@]}; do
        python align_metric_with_ransac_lidar.py \
        /home/geiger/gwb215/datasets/iphone/$seq/depth/1x \
        /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$DEPTH_METHOD/1x \
        /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/ransac_lidar_aligned_$DEPTH_METHOD/1x
    done
fi