#!/bin/bash

#dataset="dnerf"
#scenes=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")
dataset="hypernerf"
scenes=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")
#dataset="iphone"
#scenes=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "space-out" "spin" "sriracha-tree" "teddy" "wheel")


#exps=("fourbatch")
#exps=("nowarm" "noopareset" "nowopareset") 
exps=("AST" "noAST" "decoder")
#exps=("nodecoder")

#variants=("1" "2" "3")
variants=("1" "2" "3")

methods=("Curve" "FourDim" "HexPlane" "MLP" "TRBF")


for method in "${methods[@]}"; do
	for scene in "${scenes[@]}"; do
		selected_strings=()
		for exp in "${exps[@]}"; do
			for variant in "${variants[@]}"; do 
				if [ -f "output/$dataset/$scene/$method/$exp$variant/test.txt" ]; then
					selected_strings+=(" $exp$variant")
				fi
			done
		done
		echo "$method $scene ${selected_strings[@]}"
	done

done
