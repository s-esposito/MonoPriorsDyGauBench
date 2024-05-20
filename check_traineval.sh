#!/bin/bash

dataset="dnerf"
scenes=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")
#dataset="hypernerf"
#scenes=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")
#dataset="iphone"
#scenes=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "space-out" "spin" "sriracha-tree" "teddy" "wheel")
#dataset="nerfds"
#scenes=("as" "basin" "bell" "cup" "plate" "press" "sieve")
#dataset="nerfies"
#scenes=("broom" "curls" "tail" "toby-sit")


exps=("nodeform")
declare -A method_exps=(
	["Curve"]=""
	["FourDim"]=""
	["HexPlane"]=""
	["MLP"]="nodeform"
	["TRBF"]=""
)

#exps=("vanilla" "nodecoder")
#declare -A method_exps=(
#	["Curve"]="vanilla"
#	["FourDim"]="vanilla"
#	["HexPlane"]="vanilla"
#	["MLP"]="vanilla"
#	["TRBF"]="nodecoder vanilla"
#)


#exps=("onebatch" "twobatch" "fourbatch" "vanilla" "nodecoder")
#declare -A method_exps=(
#	["Curve"]="vanilla twobatch fourbatch"
#	["FourDim"]="onebatch twobatch vanilla"
#	["HexPlane"]="onebatch vanilla fourbatch"
#	["MLP"]="vanilla twobatch fourbatch"
#	["TRBF"]="nodecoder onebatch fourbatch"
#)

#exps=("nowarm" "noopareset" "nowopareset")
#declare -A method_exps=(
#	["Curve"]="nowarm noopareset nowopareset"
#	["FourDim"]="noopareset"
#	["HexPlane"]="nowarm noopareset nowopareset"
#	["MLP"]="nowarm noopareset nowopareset"
#	["TRBF"]="noopareset"
#)

#exps=("AST" "noAST" "decoder")
#declare -A method_exps=(
#  ["Curve"]="AST decoder"
#  ["FourDim"]="AST decoder"
#  ["HexPlane"]="AST decoder"
#  ["MLP"]="noAST decoder"
#  ["TRBF"]="AST"
#)

#exps=("randinit" "sfminit" "static")
#declare -A method_exps=(
#  ["Curve"]="sfminit"
#  ["FourDim"]="randinit"
#  ["HexPlane"]="randinit static"
#  ["MLP"]="randinit static"
#  ["TRBF"]="randinit"
#)

#exps=("nodeform")
#declare -A method_exps=(
#  ["MLP"]="nodeform"
#)


variants=("1" "2" "3")
methods=("Curve" "FourDim" "HexPlane" "MLP" "TRBF")
num=10
for method in "${methods[@]}"; do
  for scene in "${scenes[@]}"; do
    selected_strings=""
    for exp in ${method_exps[$method]}; do
      for variant in "${variants[@]}"; do
        if [ ! -f "output/$dataset/$scene/$method/$exp$variant/test.txt" ]; then 
          selected_strings+="$exp${variant}_incomplete "
        elif [ ! -f "output/$dataset/$scene/$method/$exp$variant/train.txt" ]; then
          selected_strings+="$exp${variant}_notrain "
        else
          test_file="output/$dataset/$scene/$method/$exp$variant/test.txt"
          train_file="output/$dataset/$scene/$method/$exp$variant/train.txt"
          
          # Extract the float number from test.txt
          test_psnr=$(grep -oP 'Average PSNR: \K[\d.]+' "$test_file")
          
          # Extract the float number from train.txt
          train_psnr=$(grep -oP 'Average PSNR: \K[\d.]+' "$train_file")
          #echo "$test_file, $train_file"
          if awk 'BEGIN {exit !("'"$test_psnr"'" > "'"$train_psnr"'")}'; then
            if (( $(echo "$test_psnr > 10" | bc -l) )); then
              echo $test_psnr
              selected_strings+="$exp$variant "
            fi
          fi
        fi
      done
    done
    printf "%-10s %-20s %s\n" "$method" "$scene" "$selected_strings"
  done
done
