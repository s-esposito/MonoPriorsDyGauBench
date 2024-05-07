import os
import sys
import yaml
import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to config file")
    parser.add_argument("--group", "-g", type=str, default="default", help="wandb group name")
    parser.add_argument("--name", "-n", type=str, help="Experiment name")
    parser.add_argument("--dataset", type=str, help="Path to the dataset")
    parser.add_argument("--slurm_script", type=str, default="slurms/dnerf_custom.sh", help="Path to slurm script")
    parser.add_argument("--output_dir", type=str, help="Output directory")

    args = parser.parse_args()

    # Load config file, yaml format
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create a new config file, replace name, group
    config["name"] = args.name
    config["group"] = args.group
    config["output"] = args.output_dir

    # Replace dataset path, which is in data -> init_args -> datadir
    config["data"]["init_args"]["datadir"] = args.dataset

    # Save the new config file
    new_config_name = os.path.join(os.path.dirname(args.config_file), f"{args.name}.yaml")
    with open(new_config_name, "w") as f:
        yaml.dump(config, f)

    print(f"New config file saved at {new_config_name}")
    print(f"Dataset path replaced with {args.dataset}")
    print(f"Experiment name: {args.name}")

    # Run the experiment
    slurm_command = f"sbatch -J {args.name} {args.slurm_script} {new_config_name} {args.output_dir}"
    print(f"Running the experiment with the following command: \n {slurm_command}")
    os.system(slurm_command)
    
