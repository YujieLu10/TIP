import pandas as pd
from icecream import ic
import argparse
import csv

parser = argparse.ArgumentParser(description='Intervention')
parser.add_argument('--load_result_path', type=str, default="", help='load result filepath')
parser.add_argument('--data_type', choices=['wikihow', 'robothow', 'wiki_inter', 'robot_inter'], default='robothow', help='choices')
parser.add_argument('--sample_for_eval', type=int, default=50)
parser.add_argument('--sample_per_worker', type=int, default=7)
parser.add_argument('--model_type', choices=['human', 'concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], default='base', help='choices')
args = parser.parse_args()

path_set = [
    "/Users/yujie/Desktop/{}/chain-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/yujie/Desktop/{}/planner-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/yujie/Desktop/{}/concept-gpt2-xl_sumFalse_task_result.txt".format(args.data_type),
    "/Users/yujie/Desktop/{}/chain-bart_sumFalse_task_result.txt".format(args.data_type),
    "/Users/yujie/Desktop/{}/planner-bart_sumFalse_task_result.txt".format(args.data_type),
    "/Users/yujie/Desktop/{}/concept-bart_sumFalse_task_result.txt".format(args.data_type),
]

path_set_to_model = {
    "/Users/yujie/Desktop/{}/chain-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_chain",
    "/Users/yujie/Desktop/{}/planner-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_planner",
    "/Users/yujie/Desktop/{}/concept-gpt2-xl_sumFalse_task_result.txt".format(args.data_type): "gpt_concept",
    "/Users/yujie/Desktop/{}/chain-bart_sumFalse_task_result.txt".format(args.data_type): "bart_chain",
    "/Users/yujie/Desktop/{}/planner-bart_sumFalse_task_result.txt".format(args.data_type): "bart_planner",
    "/Users/yujie/Desktop/{}/concept-bart_sumFalse_task_result.txt".format(args.data_type): "bart_concept",
}
with open("{}_humaneval.csv".format(args.data_type), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    head_line = ['example_idx']
    ic(args.sample_per_worker)
    for idx in range(args.sample_per_worker):
        head_line += ['program{}_idx'.format(idx+1)] + ['task{}_txt'.format(idx+1)] + ['steps{}_txt'.format(idx+1)]
    writer.writerow(head_line)

    program_idx = 0
    sample = 0    
    is_human = True
    program_list_map = {"human":[], "gpt_chain":[], "gpt_planner":[], "gpt_concept":[], "bart_chain":[], "bart_planner":[], "bart_concept":[]}
    for file_path in path_set:
        ic(file_path)
        sample = 0
        with open(file_path, 'r') as fp:
            # head line
            line = fp.readline()
            while line and sample < args.sample_for_eval: 
                if "GIVEN EXAMPLE" in line:
                    sample += 1
                    gt_line = fp.readline()
                    if is_human:
                        program_list_map["human"].append(gt_line.strip())   
                    line = fp.readline()
                    program_list_map[path_set_to_model[file_path]].append(line.strip())
                line = fp.readline()
        is_human = False


    csv_line = []
    ic(len(program_list_map["human"]))
    for example_idx in range(len(program_list_map["human"])):
        program_idx += 1
        csv_line += [str(example_idx+1)] + [str(program_idx)] + [program_list_map["human"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["human"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_chain"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_chain"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_planner"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_planner"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["gpt_concept"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["gpt_concept"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_chain"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_chain"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_planner"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_planner"][example_idx].split('.')[1:])]
        program_idx += 1
        csv_line += [str(program_idx)] + [program_list_map["bart_concept"][example_idx].split('.')[0]] + ['<br>'.join(program_list_map["bart_concept"][example_idx].split('.')[1:])]
        writer.writerow(csv_line)
        csv_line = []

# human eval result clean
# import pandas as pd
# import numpy as np

# df = pd.read_csv('/Users/yujie/Desktop/Batch_4738843_batch_results.csv')
# # df = pd.read_csv('/Users/yujie/Desktop/Batch_4738886_batch_results.csv')
# # df = pd.read_csv('/Users/yujie/Desktop/Batch_4738865_batch_results.csv')

# ic(df.keys())


# df.loc[df['Input.steps1_txt'].isnull(), 'Answer.score1'] = 1
# df.loc[df['Input.steps2_txt'].isnull(), 'Answer.score2'] = 1
# df.loc[df['Input.steps3_txt'].isnull(), 'Answer.score3'] = 1
# df.loc[df['Input.steps4_txt'].isnull(), 'Answer.score4'] = 2
# df.loc[df['Input.steps5_txt'].isnull(), 'Answer.score5'] = 1
# df.loc[df['Input.steps6_txt'].isnull(), 'Answer.score6'] = 1
# df.loc[df['Input.steps7_txt'].isnull(), 'Answer.score7'] = 2
# # df = df[df['Input.steps1_txt'].notna()]
# # df = df[df['Input.steps2_txt'].notna()]
# # df = df[df['Input.steps3_txt'].notna()]
# # df = df[df['Input.steps4_txt'].notna()]
# # df = df[df['Input.steps5_txt'].notna()]
# # df = df[df['Input.steps6_txt'].notna()]
# # df = df[df['Input.steps7_txt'].notna()]

# df.to_csv('/Users/yujie/Desktop/robothow_original.csv')

# get chain of thought for wikihow
# df = pd.read_csv('/Users/yujie/Desktop/wikisepdata/wikihowSep.csv')
# df = df.loc[df['title'].isin(['How to Take Care of a Hamster That is Giving Birth3', 'How to Disinfect a Hamster\'s Cage3', 'How to Make Hamster Health Food3', 'How to Make Baby Dwarf Hamster Food3', 'How to Treat Conjunctivitis on Guinea Pigs', 'How to Play with a Guinea Pig2', 'How to Take Care of a Turkish Angora2', 'How to Make a Squirrel Feeder', 'How to Maintain a Nice Healthy Glow', 'How to Identify an American Wirehair1'])]
# ic(df)
# # df = df[1000:2000]
# df.to_csv('/Users/yujie/Desktop/wikisepdata/wikihowSep_selected.csv')
