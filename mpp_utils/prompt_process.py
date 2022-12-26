from itertools import islice
from icecream import ic

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_prompt(opt):
    batch_size = opt.n_samples
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        task_start_idx_list = []
        print(f"reading prompts from {opt.from_file}")
        if opt.file_type == "json":
            import json
            with open('/local/home/yujielu/project/MPP/submodules/GoalAgent/data/{}/{}_available_examples.json'.format(opt.data_type, opt.data_type), 'r') as f:
                available_examples = json.load(f)[:opt.task_num]
            # example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
            # example_task_embedding = translation_lm.encode(example_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
            summarize_example_data_list = []
            data = []
            idx = 0
            for example in available_examples:
                task_start_idx_list.append(idx)
                idx += 1
                data.append(example.split('\n')[0])
                for step in example.split('\n')[1:]:
                    idx += 1
                    data.append(step)
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
                    ic(task_start_idx_list)
                    for item in new_p:
                        idx += 1
                        if len(item) > 1:
                            data.append(item)
        ic(len(data))
        with open('prompt_list_{}_{}.txt'.format(opt.data_type, opt.task), 'w') as f:
            for line in data:
                f.write(f"{line}\n")
        data = list(chunk(data, batch_size))
    return data, task_start_idx_list, summarize_example_data_list