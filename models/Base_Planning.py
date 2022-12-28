class Base_Planner(object):
    def __init__(self, opt):
        self.total_score_cal = {}
        method_type=opt.model_type
        all_model_type_list = ['{}'.format(method_type)]
        for model_type_item in all_model_type_list:
            self.total_score_cal[model_type_item] = {"sentence-bleu": 0, "wmd": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0, "meteor": 0, "sentence-bert-score": 0, "caption-t-bleu": 0, "LCS": 0, "caption-t-bleu": 0, "caption-vcap-lcs": 0, "gpt3-plan-accuracy": 0, "caption-gpt3-plan-accuracy": 0, "vplan-t-clip-score": 0, "tplan-v-clip-score": 0, "vplan-v-clip-score": 0, "tplan-t-clip-score": 0}