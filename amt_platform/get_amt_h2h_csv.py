import os
import pandas as pd
from icecream import ic
import argparse
import csv
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--root_path', type=str, default=f"/share/edc/home/yujielu/MPP_data/all_plan_grid", help='image root')
    parser.add_argument('--amt_root_path', type=str, default=f"https://mpp-ig.s3.us-west-1.amazonaws.com/", help='image root')
    parser.add_argument('--ours_model', type=str, default="u-plan-bridge", help='our model')
    parser.add_argument('--source', type=str, default="groundtruth_input", help='source')
    parser.add_argument('--eval_task', type=str, default="all", help='eval task')
    parser.add_argument('--load_result_path', type=str, default="", help='load result filepath')
    parser.add_argument('--data_type', choices=['wikihow', 'recipeqa', 'all'], default='wikihow', help='choices')
    parser.add_argument('--sample_for_eval', type=int, default=5)
    # parser.add_argument('--sample_per_worker', type=int, default=1)
    parser.add_argument('--model_type', choices=['human', 'concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], default='base', help='choices')
    opt = parser.parse_args()
    return opt

def generate_batch_csv(opt, path_dir):
    with open(f"amt_platform/data/{opt.data_type}_amt_h2h.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        head_line = ['example_idx'] + ['datatype'] + ['task_txt']
        sample_per_worker = 2
        for idx in range(sample_per_worker):
            head_line += ['program{}_idx'.format(idx+1)] + ['plan{}_grid_path'.format(idx+1)]
        writer.writerow(head_line)

        data_type_list = ["wikihow", "recipeqa"] if opt.data_type == "all" else opt.data_type
        program_list_map = {}
        for comparison_base in path_dir:
            plan_grid_num = opt.sample_for_eval # len(glob.glob1(comparison_base["path"], "plan_grid_task_[0-9]*.png"))
            for datatype in data_type_list:
                for plan_idx in range(plan_grid_num):
                    if comparison_base["exp"] == opt.ours_model:
                        imgpath = opt.amt_root_path + f"exp/all_plan_grid/{opt.source}/{datatype}/u-plan/" + f"plan_grid_task_{plan_idx}_bridge.png"
                    else:
                        basename = comparison_base["exp"]
                        imgpath = opt.amt_root_path + f"exp/all_plan_grid/{opt.source}/{datatype}/{basename}/" + f"plan_grid_task_{plan_idx}.png"
                    if not comparison_base["exp"] in program_list_map.keys():
                        program_list_map[comparison_base["exp"]] = []
                    program_list_map[comparison_base["exp"]].append(imgpath)

        example_idx = 0
        program_idx = 0
        for datatype in data_type_list:
            for task_sample_idx in range(opt.sample_for_eval):
                with open(os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", datatype, f"task_{task_sample_idx}", "task.txt"), "r") as ftask:
                    task_name = ftask.readline()
                for comparison_base in path_dir:
                    if comparison_base["exp"] == opt.ours_model: continue
                    example_idx += 1   
                    csv_line = [str(example_idx)] + [datatype] + [task_name]
                    program_idx += 1
                    csv_line += [str(program_idx)] + [program_list_map[path_dir[0]["exp"]][task_sample_idx]]
                    program_idx += 1
                    csv_line += [str(program_idx)] + [program_list_map[comparison_base["exp"]][task_sample_idx]]
                    writer.writerow(csv_line)
                    csv_line = []


if __name__ == "__main__":
    opt = parse_args()
    data_root = os.path.join(opt.root_path, opt.source, opt.data_type)
    path_dir = [{"exp": opt.ours_model, "path": os.path.join(data_root, "u-plan")}]
    # os.listdir(data_root)
    for exp_name in ["c-plan", "u-plan", "tgt-u-plan", "dalle", "vgt-u-plan"]: # experiment for comparison
        # if exp_name in ["m-plan", "all_metrics.csv"]: continue
        path_dir.append({"exp": exp_name, "path": os.path.join(data_root, exp_name)})
    generate_batch_csv(opt, path_dir)
