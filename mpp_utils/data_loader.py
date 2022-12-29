from itertools import islice
from icecream import ic
import os
from tqdm import tqdm
import glob

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_sample(opt, config, load_task=False):
    batch_size = opt.n_samples
    data = []
    task_start_idx_list = []
    summarize_example_data_list = []
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        # if config.mpp_model.task_config.ground_truth_modality == "textual":
        if opt.task in ["tgt-u-plan", "m-plan"] or load_task:
            print(f"reading prompts from {opt.from_file}")
            if opt.file_type == "json":
                import json
                with open('/local/home/yujielu/project/MPP/submodules/GoalAgent/data/{}/{}_available_examples.json'.format(opt.data_type, opt.data_type), 'r') as f:
                    available_examples = json.load(f)
                    if opt.task_num > 0: available_examples = available_examples[:opt.task_num]
                # example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
                # example_task_embedding = translation_lm.encode(example_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
                data = []
                idx = 0
                for example in available_examples:
                    task_start_idx_list.append(idx)
                    for step in example.split('\n'):
                        if opt.use_task_hint and idx > 0:
                            step = "The task is {} {}".format(data[0][6:], step)
                        data.append(step)
                        idx += 1
                    summarize_example_data_list.append({"tasks": example.split('\n')[0], "steps": '.'.join(example.split('\n')[1:])})
            else:
                with open(opt.from_file, "r") as f:
                    data = f.read().splitlines()
                    data = [p for p in data for i in range(opt.repeat)]
                    old_data = data.copy()
                    data = []
                    idx = 0
                    for p in old_data:
                        new_p = p.split("##STEP##")
                        task_start_idx_list.append(idx)
                        idx += 1
                        for item in new_p:
                            idx += 1
                            if len(item) > 1:
                                data.append(item)
        elif opt.task in ["u-plan"]:
            resolution_config = "resolution_{}".format(opt.resolution)
            text_path = os.path.join(opt.outdir, "debug_output" if opt.debug else "experiment_output", resolution_config, opt.data_type, "bridge" if opt.use_bridge else "origin", opt.task+("_w_task_hint" if opt.use_task_hint else "")) # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
            task_num = len(os.listdir(text_path))
            all_step_count = 0
            for task_idx in tqdm(range(task_num)):
                data = []
                sample_path = os.path.join(text_path, f"task_{str(task_idx)}")
                if not os.path.exists(sample_path):
                    continue
                task_start_idx_list.append(all_step_count)
                # step_num = len(os.listdir(sample_path))
                step_num = len(glob.glob1(sample_path,"step_*.txt"))
                for step_idx in range(step_num):
                    all_step_count += 1
                    with open(os.path.join(sample_path, f"step_{step_idx}.txt"), 'r') as f:
                        current_text = f.readline()
                        if opt.use_task_hint and step_idx > 0:
                            current_text = "The task is {} {}".format(data[0][6:], current_text)
                        data.append(current_text)
                summarize_example_data_list.append({"tasks": data[0], "steps": data[1:]})
        data = list(chunk(data, batch_size))
    return data, task_start_idx_list, summarize_example_data_list