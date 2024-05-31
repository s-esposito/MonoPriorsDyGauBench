#!/bin/bash

#dataset="dnerf"
#scenes=("bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex")

#path="HyperNeRF_Scenes_Motion_Masks"
#dataset="hypernerf"
#scenes=("aleks-teapot" "americano" "broom2" "chickchicken" "cross-hands1" "cut-lemon1" "espresso" "hand1-dense-v2" "keyboard" "oven-mitts" "slice-banana" "split-cookie" "tamping" "torchocolate" "vrig-3dprinter" "vrig-chicken" "vrig-peel-banana")

path="motion_iphone"
dataset="iphone"
scenes=("apple" "backpack" "block" "creeper" "handwavy" "haru-sit" "mochi-high-five" "paper-windmill" "pillow" "space-out" "spin" "sriracha-tree" "teddy" "wheel")
#dataset="nerfds"
#scenes=("as" "basin" "bell" "cup" "plate" "press" "sieve")
#dataset="nerfies"
#scenes=("broom" "curls" "tail" "toby-sit")

#exps=("vanilla" "nodecoder")
#declare -A method_exps=(
#	["Curve"]="vanilla "
#	["FourDim"]="vanilla "
#	["HexPlane"]="vanilla "
#	["MLP"]="vanilla "
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

cd $path

for scene in "${scenes[@]}"; do
  if [ ! -e ../data/$dataset/$scene/resized_mask ]; then
    echo $scene
    unzip $scene.zip
    mv mask-colmap ../data/$dataset/$scene/resized_mask
  fi
done

cd ..