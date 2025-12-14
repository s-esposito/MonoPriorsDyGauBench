datasets=(
    # dnerf
    # fixed
    # hypernerf
    # nerfds
    iphone
    # nerfies
)

nerfies_scenes=(
    broom
    curls
    tail
    toby-sit
)

nerfds_scenes=(
    as
    basin
    bell
    # cup
    # plate
    # press
    # sieve
)

iphone_scenes=(
    # apple
    backpack
    # block
    # creeper
    # handwavy
    # haru-sit
    # mochi-high-five
    # paper-windmill
    # pillow
    # space-out
    # spin
    # sriracha-tree
    # teddy
    # wheel
)

methods=(
    # Curve
    # HexPlane
    MLP
)

group_name="depth_experiment"
depth_methods=("videoda") # "mega-sam" "depth-pro")    # "depth-pro", "videoda", "mega-sam"
lambda_depth=0.01           # choose nerfies: 0.05; d-nerf: 0.005
type="DepthSupervision-" # "nowarmuptestWmodelinit0.25DepthSfMAlign-" #"nowarmuptestDepthSfMAlign-" OldInitTestDepthSupervision RansacInitTestDepthSupervision
# Debug Runs: "ScaleDeebugDepthSfMAlign", "DeebugDepthSfMAlign-", "DeebugAbsdClampScale-", "DepthMapPoints2DeebugDepthSfMAlign-"
# Real Runs: "LSICPProcrustesFGDepthSfMAlign", "LS2dDepthSfMAlign-", "ICPDepthSfMAlign-", "DecayingInvAlignLossAbsScale-", "0.01InvAlignLoss", "new-depthrendering-LogSSI-"
# model.init_args.lambda_depth ${lambda_depth}
compute_mask_test=true


for depth_method in "${depth_methods[@]}"; do
    for method in "${methods[@]}"; do
        for dataset in "${datasets[@]}"; do
            declare -n scenes_ref="${dataset}_scenes"
            scenes=("${scenes_ref[@]}")
            for scene in "${scenes[@]}"; do
                base="${dataset}/${scene}/${method}"
                name="vanilla1"
                variant="${base}/${name%?}1"
                output_path="./output/depth_experiment/${base}-${type}${depth_method}"

                ### TRAIN ###
                echo "Processing method ${method} with dataset ${dataset} on scene ${scene} with variant ${variant} and output path ${output_path}"
                python main.py fit \
                    --config configs/${variant}.yaml \
                    --output ${output_path} \
                    --name "${scene}-${base##*/}-${type}${depth_method}_$name" \
                    --group "${group_name}" \
                    --data.init_args.depth_method "${depth_method}" \
                    --data.init_args.num_pts 0 \

                ### TEST ###
                python main.py test \
                    --config configs/${variant}.yaml \
                    --ckpt_path last \
                    --output ${output_path} \
                    --name "${scene}-${base##*/}-${type}${depth_method}_$name" \
                    --group "${group_name}" \
                    --data.init_args.eval_train False \
                    --data.init_args.depth_method "${depth_method}" \
                # TEST ON TRAIN SET #
                python main.py test \
                    --config configs/${variant}.yaml \
                    --ckpt_path last \
                    --output ${output_path} \
                    --name "${scene}-${base##*/}-${type}${depth_method}_${name}_train" \
                    --group "${group_name}" \
                    --data.init_args.eval_train True \
                    --data.init_args.depth_method "${depth_method}" \
                
                ### MASKED TEST (OPTIONAL) ###
                if [[ "$compute_mask_test" == true ]]; then
                    python main.py test \
                        --config configs/${variant}.yaml \
                        --ckpt_path last \
                        --output ${output_path} \
                        --name "${scene}-${base##*/}-${type}${depth_method}_${name}_masked" \
                        --group "${group_name}" \
                        --data.init_args.eval_train False \
                        --data.init_args.depth_method "${depth_method}" \
                        --model.init_args.eval_mask True \
                        --data.init_args.load_mask True
                    # TEST ON TRAIN SET #
                    python main.py test \
                        --config configs/${variant}.yaml \
                        --ckpt_path last \
                        --output ${output_path} \
                        --name "${scene}-${base##*/}-${type}${depth_method}_${name}_train_masked" \
                        --group "${group_name}" \
                        --data.init_args.eval_train True \
                        --data.init_args.depth_method "${depth_method}" \
                        --model.init_args.eval_mask True \
                        --data.init_args.load_mask True 
                fi
            done
        done
    done
done