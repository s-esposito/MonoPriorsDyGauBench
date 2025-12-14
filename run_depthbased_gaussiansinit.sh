datasets=(
    # dnerf
    # fixed
    # hypernerf
    # nerfds
    # nerfies
    iphone
)

nerfies_scenes=(
    # broom
    # curls
    # tail
    toby-sit
)

iphone_scenes=(
#     apple
#     backpack
#     block
#     creeper
#     handwavy
#     haru-sit
#     mochi-high-five
#     paper-windmill
    pillow
#     # space-out
#     spin
#     sriracha-tree
#     teddy
#     # wheel
)

methods=(
    Curve
    # HexPlane
    # MLP
)

group_name="depth_experiment"
depth_method="mega-sam"    # "depth-pro", "videoda", "mega-sam"
lambda_depth=0.01
loss_function="DepthSupervision+GaussianInit-"
compute_mask_test=true

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
            output_path="./output/depth_experiment/${base}-${loss_function}${depth_method}"


            echo "Processing method ${method} with dataset ${dataset} on scene ${scene} with variant ${variant} and output path ${output_path}"
            python main.py fit \
                --config configs/${variant}.yaml \
                --output ${output_path} \
                --name "${scene}-${base##*/}-${loss_function}${depth_method}_$name" \
                --group "${group_name}" \
                --data.init_args.depth_method "${depth_method}" \
                --data.init_args.num_pts 0 \
                --model.init_args.warm_up 0 \


            python main.py test \
                --config configs/${variant}.yaml \
                --ckpt_path last \
                --output ${output_path} \
                --name "${scene}-${base##*/}-${loss_function}${depth_method}_$name" \
                --group "${group_name}" \
                --data.init_args.eval_train False \
                --data.init_args.depth_method "${depth_method}" \

            python main.py test \
                --config configs/${variant}.yaml \
                --ckpt_path last \
                --output ${output_path} \
                --name "${scene}-${base##*/}-${loss_function}${depth_method}_${name}_train" \
                --group "${group_name}" \
                --data.init_args.eval_train True \
                --data.init_args.depth_method "${depth_method}" \
            
            # extra test runs if compute_mask_test is true
            if [[ "$compute_mask_test" == true ]]; then
                python main.py test \
                    --config configs/${variant}.yaml \
                    --ckpt_path last \
                    --output ${output_path} \
                    --name "${scene}-${base##*/}-${loss_function}${depth_method}_${name}_masked" \
                    --group "${group_name}" \
                    --data.init_args.eval_train False \
                    --data.init_args.depth_method "${depth_method}" \
                    --model.init_args.eval_mask True \
                    --data.init_args.load_mask True

                python main.py test \
                    --config configs/${variant}.yaml \
                    --ckpt_path last \
                    --output ${output_path} \
                    --name "${scene}-${base##*/}-${loss_function}${depth_method}_${name}_train_masked" \
                    --group "${group_name}" \
                    --data.init_args.eval_train True \
                    --data.init_args.depth_method "${depth_method}" \
                    --model.init_args.eval_mask True \
                    --data.init_args.load_mask True 
            fi
        done
    done
done