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

# For preproc unidepth aligned depth anything
for seq in ${scenes[@]}; do
    python convert_iphone_disp_to_depth.py \
    /home/geiger/gwb215/datasets/iphone/$seq/aligned_depth_anything_v2/1x \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/metric_aligned_depth_anything_v2/1x
done

# # For colmap aligned depth anything
# for seq in ${scenes[@]}; do
#     python convert_iphone_disp_to_depth.py \
#     /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/aligned_depth_anything_colmap/1x \
#     /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/metric_aligned_depth_anything_colmap_depth/1x
# done

