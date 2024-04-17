#!/bin/bash

data_dir=$1
dnerf_dir="${data_dir}/dnerf"
nerfies_dir="${data_dir}/nerfies"
hypernerf_dir="${data_dir}/hypernerf"
nerfds_dir="${data_dir}/nerfds"
iphone_dir="${data_dir}/iphone"


# Check gdown version
installed_version=$(gdown --version | awk '{print $2}')
required_version="4.7.3"

if [[ "$installed_version" != "$required_version" ]]; then
    echo "Error: gdown version $required_version is required, but version $installed_version is installed."
    echo "Please install gdown version $required_version and run the script again."
    exit 1
fi

# D-NeRF
if [ ! -d "${dnerf_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${dnerf_dir}"
    echo "Directory ${dnerf_dir} created."
    cd ${dnerf_dir}
    wget https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip
    unzip data.zip
    rm data.zip
else
    echo "Directory ${dnerf_dir} already exists."
fi

# NeRF-DS
if [ ! -d "${nerfds_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${nerfds_dir}"
    echo "Directory ${nerfds_dir} created."
    cd ${nerfds_dir}
    wget https://github.com/JokerYan/NeRF-DS/releases/download/v0.1-pre-release/NeRF-DS.dataset.zip
    unzip "NeRF-DS.dataset.zip"
    rm "NeRF-DS.dataset.zip"


else
    echo "Directory ${nerfds_dir} already exists."
fi


# Iphone
if [ ! -d "${iphone_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${iphone_dir}"
    echo "Directory ${iphone_dir} created."
fi

cd ${iphone_dir}

iphone_scenes="apple backpack block creeper handwavy haru-sit mochi-high-five paper-windmill pillow space-out spin sriracha-tree teddy wheel"
iphone_ids="1vlT4ie1NYsv_po0Oxen0kCKkR1Q_ARgC 1vzm50Jsbkdv1ilpqS_eEs8d5QglTxm-s 1k_SmsXFvdbQGlwfWcHErAPmxEqxFZUhg 1FeUqbB000RrZLDNH8dxlm6Ak6Kd7FaIS 1QaqMe6cGdN_HF4jEsLpIdOaes-oUQGgX 1_svZDXhqjVPUIf3iInQHO0JWerFKQbOP 1PUVGyGwRNlw9Opd6sjhUk3NLaRsuFiJH 1n32orTLJ5IbwbyPewgbu6Bqj9M9kAzjj 1DIWuNYNjlTEUqoiQw33aBzhBZV2Z4YfK 1gvX9dzigKMTM_IkhjVTQ7770JURxCiBv 1rKbgffouLf_TAaBMk78G-yfPJFYFGjfH 1SYkq5cfoweEUIo91ExFR3yYeC4E3Gqxv 1TduWcT24Jlpq7GOeU2Dtv4mD-2D1I8RT 1aZtlwaKuF4g7i7u5UG5hMjVv0pBy42o4"

# Convert the space-separated lists into arrays
IFS=' ' read -r -a scenes_array <<< "$iphone_scenes"
IFS=' ' read -r -a ids_array <<< "$iphone_ids"

for ((i=0; i<${#scenes_array[@]}; i++))
do
    scene="${scenes_array[$i]}"
    id="${ids_array[$i]}"


    echo "Scene: $scene, ID: $id" 
    if [ ! -d "./${scene}" ]; then
        gdown $id
        unzip "${scene}.zip"
        rm "${scene}.zip"
    else
        echo "Directory ${iphone_dir}/${scene} already exists."
    fi
done

cd ../..

# Nerfies
echo pwd 

if [ ! -d "${nerfies_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${nerfies_dir}"
    echo "Directory ${nerfies_dir} created."
    cd ${nerfies_dir}
    wget https://github.com/google/nerfies/releases/download/0.1/nerfies-vrig-dataset-v0.1.zip
    unzip "nerfies-vrig-dataset-v0.1.zip"
    rm "nerfies-vrig-dataset-v0.1.zip"

else
    echo "Directory ${nerfies_dir} already exists."
fi


# HyperNeRF
if [ ! -d "${hypernerf_dir}" ]; then
    # Create the directory if it doesn't exist
    mkdir -p "${hypernerf_dir}"
    echo "Directory ${hypernerf_dir} created."
fi

cd ${hypernerf_dir}

hypernerf_scenes="interp_aleks-teapot interp_chickchicken interp_cut-lemon interp_hand interp_slice-banana interp_torchocolate misc_americano misc_cross-hands misc_espresso misc_keyboard misc_oven-mitts misc_split-cookie misc_tamping vrig_3dprinter vrig_broom vrig_chicken vrig_peel-banana"
#1vlT4ie1NYsv_po0Oxen0kCKkR1Q_ARgC 1vzm50Jsbkdv1ilpqS_eEs8d5QglTxm-s 1k_SmsXFvdbQGlwfWcHErAPmxEqxFZUhg 1FeUqbB000RrZLDNH8dxlm6Ak6Kd7FaIS 1QaqMe6cGdN_HF4jEsLpIdOaes-oUQGgX 1_svZDXhqjVPUIf3iInQHO0JWerFKQbOP 1PUVGyGwRNlw9Opd6sjhUk3NLaRsuFiJH 1n32orTLJ5IbwbyPewgbu6Bqj9M9kAzjj 1DIWuNYNjlTEUqoiQw33aBzhBZV2Z4YfK 1gvX9dzigKMTM_IkhjVTQ7770JURxCiBv 1rKbgffouLf_TAaBMk78G-yfPJFYFGjfH 1SYkq5cfoweEUIo91ExFR3yYeC4E3Gqxv 1TduWcT24Jlpq7GOeU2Dtv4mD-2D1I8RT 1aZtlwaKuF4g7i7u5UG5hMjVv0pBy42o4"
link="https://github.com/google/hypernerf/releases/download/v0.1/"
# Convert the space-separated lists into arrays
IFS=' ' read -r -a scenes_array <<< "$hypernerf_scenes"
#IFS=' ' read -r -a links_array <<< "$hypernerf_links"

for ((i=0; i<${#scenes_array[@]}; i++))
do
    scene="${scenes_array[$i]}"
    id="${link}/${scene}.zip"
    scenename="${scene#*_}"


    echo "Scene: $scene, link: $id, scenename: $scenename" 
    if [ ! -d "./${scenename}"* ]; then
        wget $id
        unzip "${scene}.zip"
        rm "${scene}.zip"
    else
        echo "Directory ${hypernerf_dir}/${scene} already exists."
    fi
done

cd ../..




# Our Custom
