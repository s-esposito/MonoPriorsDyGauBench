#!/bin/bash

#datasets=("dnerf" "hypernerf" "nerfds" "nerfies")

datasets=("dnerf/bouncingballs" "dnerf/hellwarrior" "dnerf/hook" "dnerf/jumpingjacks" "dnerf/lego" "dnerf/mutant" "dnerf/standup" "dnerf/trex" "hypernerf/aleks-teapot" "hypernerf/americano" "hypernerf/broom2" "hypernerf/chickchicken" "hypernerf/cross-hands1" "hypernerf/cut-lemon1" "hypernerf/espresso" "hypernerf/hand1-dense-v2" "hypernerf/keyboard" "hypernerf/oven-mitts" "hypernerf/slice-banana" "hypernerf/split-cookie" "hypernerf/tamping" "hypernerf/torchocolate" "hypernerf/vrig-3dprinter" "hypernerf/vrig-chicken" "hypernerf/vrig-peel-banana" "nerfds/as" "nerfds/basin" "nerfds/bell" "nerfds/cup" "nerfds/plate" "nerfds/press" "nerfds/sieve" "nerfies/broom" "nerfies/curls" "nerfies/tail" "nerfies/toby-sit" "iphone/apple" "iphone/backpack" "iphone/block" "iphone/creeper" "iphone/handwavy" "iphone/haru-sit" "iphone/mochi-high-five" "iphone/paper-windmill" "iphone/pillow" "iphone/space-out" "iphone/spin" "iphone/sriracha-tree" "iphone/teddy" "iphone/wheel")

exps=("Curve/vanilla" "FourDim/vanilla" "HexPlane/vanilla" "MLP/vanilla" "TRBF/nodecoder" "TRBF/vanilla")
#exps=("Curve/twobatch" "Curve/fourbatch" "FourDim/onebatch" "FourDim/twobatch" "HexPlane/onebatch" "HexPlane/fourbatch" "MLP/twobatch" "MLP/fourbatch" "TRBF/onebatch" "TRBF/fourbatch")
#exps=("decoder.sh" "AST.sh" "noAST.sh")

variants=("1" "2" "3")

for dataset in "${datasets[@]}"; do
    for exp in "${exps[@]}"; do
        for variant in "${variants[@]}"; do
            target_path="output/$dataset/$exp$variant" 
            if [ ! -f $target_path/test.txt ]; then
                if [ ! -d $target_path ]; then
                    mkdir $target_path
                fi
                echo "$target_path <- yiqingl@scdt.stanford.edu:/orion/u/yiqingl/GaussianDiff/$target_path"
                scp yiqingl@scdt.stanford.edu:/orion/u/yiqingl/GaussianDiff/$target_path/test.txt $target_path/test.txt
            fi
        done
    done
done