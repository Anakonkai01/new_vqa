import os
import subprocess
import argparse

# Ablation matrix according to Model_G_Architecture_Specification_v2.md
# 10 experiments: 0 = Baseline, 1-8 = feature subsets, 9 = Full Model G
EXPERIMENTS = {
    0: {"name": "0_baseline", "flags": ""},
    1: {"name": "1_G5_len_cond", "flags": "--len_cond"},
    2: {"name": "2_G1_geo7", "flags": "--geo7"},
    3: {"name": "3_G2_pgn3", "flags": "--pgn3"},
    4: {"name": "4_G3_infonce", "flags": "--infonce"},
    5: {"name": "5_G4_ohp", "flags": "--ohp"},
    6: {"name": "6_G5_G2_len_pgn3", "flags": "--len_cond --pgn3"},
    7: {"name": "7_G5_G3_len_infonce", "flags": "--len_cond --infonce"},
    8: {"name": "8_G2_G3_G4_arch_only", "flags": "--pgn3 --infonce --ohp"},
    9: {"name": "9_Full_Model_G", "flags": "--geo7 --pgn3 --infonce --ohp --len_cond"}
}

def run_experiment(exp_id: int, base_flags: str, skip_eval: bool):
    exp = EXPERIMENTS[exp_id]
    name = exp["name"]
    flags = exp["flags"]
    
    print(f"\n{'='*60}")
    print(f"Starting Experiment {exp_id}: {name}")
    print(f"Flags: {flags}")
    print(f"{'='*60}\n")
    
    # Base configuration: ensure Model G dispatch and WandB run naming
    # Default 4 phases loop can be approximated through a shell script or train.py calls
    # For full 4 phases, we just call the shell script passing the extra flags.
    
    # Actually, we will just call train.py --model G directly for a specific phase,
    # or the bash script train_model_g.sh with flags.
    # We will assume train_model_g.sh forwards its arguments.
    
    cmd = f"bash train_model_g.sh all {base_flags} {flags} --wandb_run_name ablation_{name}"
    
    print(f"RUNNING: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment {exp_id} failed!")
        return False
        
    if not skip_eval:
        print(f"\n--- Evaluating Experiment {exp_id} ---")
        eval_cmd = f"python src/evaluate.py --model_type G --checkpoint checkpoints/model_g_best.pth --beam_width 3"
        print(f"RUNNING: {eval_cmd}")
        try:
            subprocess.run(eval_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"\n[ERROR] Evaluation for {exp_id} failed!")
            
    return True

def main():
    parser = argparse.ArgumentParser(description="Model G Ablation Runner")
    parser.add_argument('--exp', type=int, default=None, choices=list(EXPERIMENTS.keys()),
                        help="Run specific experiment ID (0-9)")
    parser.add_argument('--base_flags', type=str, default="",
                        help="Base flags to pass to all experiments (e.g. data paths, batch size)")
    parser.add_argument('--skip_eval', action='store_true',
                        help="Skip evaluation after training")
    args = parser.parse_args()

    if args.exp is not None:
        run_experiment(args.exp, args.base_flags, args.skip_eval)
    else:
        print("Running all 10 ablation experiments sequentially...")
        for i in range(10):
            success = run_experiment(i, args.base_flags, args.skip_eval)
            if not success:
                print("Stopping ablation suite due to failure.")
                break
                
if __name__ == "__main__":
    main()
