#!/bin/bash

dataset="dnerf"
scenes=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")

exps=("fourbatch")
exps=("nowarm" "noopareset" "nowopareset") 
exps=("AST" "noAST" "decoder")

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
