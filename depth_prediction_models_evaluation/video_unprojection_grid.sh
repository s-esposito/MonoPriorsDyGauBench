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

max_jobs=8  # control parallelism (e.g. 4 scenes at once)
running_jobs=0

for seq in "${scenes[@]}"; do
    (
        echo "Creating visualization images of depth unprojection comparison of Scene: $seq"
        python video_unprojection_grid.py --scene "$seq" --sparsity 47

        echo "Creating video from images"
        ffmpeg -y -framerate 25 -i depth_unprojection_comparisons/${seq}/0_%05d.png -c:v mpeg4 -q:v 15 -pix_fmt yuv420p depth_unprojection_comparisons/video/${seq}.mp4

        # 1. Generate a color palette from the video
        ffmpeg -y -i depth_unprojection_comparisons/video/${seq}.mp4 -vf "fps=30,scale=iw:-1:flags=lanczos,palettegen" -frames:v 1 ${seq}_palette.png

        # 2. Use the palette to generate the gif from the video
        ffmpeg -y -i depth_unprojection_comparisons/video/${seq}.mp4 -i ${seq}_palette.png \
            -filter_complex "fps=30,scale=iw:-1:flags=lanczos[x];[x][1:v]paletteuse" \
            depth_unprojection_comparisons/gif/${seq}.gif
    ) &

    ((running_jobs+=1))
    if [[ $running_jobs -ge $max_jobs ]]; then
        wait
        running_jobs=0
    fi
done

wait  # wait for any remaining jobs to finish