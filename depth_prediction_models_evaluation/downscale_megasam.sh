scenes=(
    # apple
    backpack
    block
    creeper
    handwavy
    haru-sit
    mochi-high-five
    paper-windmill
    pillow
    # space-out
    spin
    sriracha-tree
    teddy
    # wheel
)

for seq in ${scenes[@]}; do
    python downscale_megasam.py \
    /home/geiger/gwb215/datasets/iphone/$seq/flow3d_preprocessed/mega_sam/1x/ \
    /home/geiger/gwb215/MonoPriorsDyGauBench/data/iphone/$seq/rgb/2x_mega-sam/
done