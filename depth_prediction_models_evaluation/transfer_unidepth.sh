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
echo "Transferring UniDepth outputs to flow3d_preprocessed/unidepth2/1x for scenes: ${scenes[@]}"
for seq in ${scenes[@]}; do
    python transfer_unidepth.py \
    /home/geiger/gwb215/mega-sam/UniDepth/outputs/$seq \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/unidepth2/1x
done