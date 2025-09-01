datasets=(
    # dnerf
    # fixed
    # hypernerf
    # iphone
    # nerfds
    nerfies
)

nerfies_scenes=(
    #broom
    #curls
    #tail
    toby-sit
)

methods=(
#Curve
    # FourDim
#HexPlane
    MLP
    # TRBF
)

group_name="depth_experiment"

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        # scene_var_name="${dataset}_scenes"
        # scenes=("${!scene_var_name}")
        declare -n scenes_ref="${dataset}_scenes"
        scenes=("${scenes_ref[@]}")
        for scene in "${scenes[@]}"; do
            base="${dataset}/${scene}/${method}"
            name="vanilla1"
            variant="${base}/${name%?}1"
            output_path="./output/depth_experiment/${base}"
            # print scenes
            # echo "scenes: ${scenes[@]}"
            # echo "Processing method ${method} with dataset ${dataset} on scene ${scene} with variant ${variant} and output path ${output_path}"
            echo "Processing method ${method} with dataset ${dataset} on scene ${scene} with variant ${variant} and output path ${output_path}"
            python main.py fit \
                --config configs/${variant}.yaml \
                --output ${output_path} \
                --name "${base##*/}_$name" \
                --group "${group_name}" #\
                #--data.init_args.depth_method "depth_pro"

            python main.py test \
                --config configs/${variant}.yaml \
                --ckpt_path  last \
                --output ${output_path} \
                --name "${base##*/}_$name" \
                --group "${group_name}" \
                --data.init_args.eval_train False #\
                #--data.init_args.depth_method "depth_pro"

            python main.py test \
                --config configs/${variant}.yaml \
                --ckpt_path  last \
                --output ${output_path} \
                --name "${base##*/}_$name" \
                --group "${group_name}" \
                --data.init_args.eval_train True #\
                #--data.init_args.depth_method "depth_pro"
        done
    done
done

