method=MLP      # Curve, MLP, HexPlane
dataset=nerfies   # dnerf, fixed, hypernerf, iphone, nerfds, nerfies
scene=curls

name="vanilla2"
group_name="vis_script_test" # "depth_test_curls_utils_loss"

base="${dataset}/${scene}/${method}"
variant="${base}/${name%?}1"

output_path="./output/vis_script_test/${base}"

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