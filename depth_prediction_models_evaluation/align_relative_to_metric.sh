# Script to align relative depth with metric depth using mean scale and shift
# Calling align_metric_with_mean_scsh_lidar.py because it doesn't matter if we use lidar or metric depth to align to

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

REL_DEPTH_METHOD=depth_pro 
ABS_DEPTH_METHOD=video_depth_anything

# Call different scripts based on the depth method
# for seq in ${scenes[@]}; do
#     echo "Aligning relative depth with metric depth for sequence: $seq"
#     python align_metric_with_mean_scsh_lidar.py \
#     /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$ABS_DEPTH_METHOD/1x \
#     /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$REL_DEPTH_METHOD/1x \
#     /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/${ABS_DEPTH_METHOD}_aligned_${REL_DEPTH_METHOD}/1x
# done

# Call different scripts based on the depth method
for seq in ${scenes[@]}; do
    echo "Aligning relative depth with metric depth for sequence: $seq"
    python align_metric_with_lidar.py \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$ABS_DEPTH_METHOD/1x \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/$REL_DEPTH_METHOD/metric/1x \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/${ABS_DEPTH_METHOD}_aligned_${REL_DEPTH_METHOD}/1x
done