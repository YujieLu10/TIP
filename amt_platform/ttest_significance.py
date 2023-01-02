# https://www.statskingdom.com/paired-t-test-calculator.html
import pandas as pd
from icecream import ic
df1 = pd.read_csv('/local/home/yujielu/project/GoalAgent/causal_planner_human_eval/robot_inter_metric_correlation.csv')
df2 = pd.read_csv('/local/home/yujielu/project/GoalAgent/causal_planner_human_eval/robothow_metric_correlation.csv')
df3 = pd.read_csv('/local/home/yujielu/project/GoalAgent/causal_planner_human_eval/wiki_inter_metric_correlation.csv')
df4 = pd.read_csv('/local/home/yujielu/project/GoalAgent/causal_planner_human_eval/wikihow_metric_correlation.csv')

list1 = df1.loc[df1['method_type'] == 'gpt-concept']["human_plan"].tolist()
list2 = df2.loc[df1['method_type'] == 'gpt-concept']["human_plan"].tolist()
list3 = df3.loc[df1['method_type'] == 'gpt-concept']["human_plan"].tolist()
list4 = df4.loc[df1['method_type'] == 'gpt-concept']["human_plan"].tolist()
list5 = df1.loc[df1['method_type'] == 'bart-concept']["human_plan"].tolist()
list6 = df2.loc[df1['method_type'] == 'bart-concept']["human_plan"].tolist()
list7 = df3.loc[df1['method_type'] == 'bart-concept']["human_plan"].tolist()
list8 = df4.loc[df1['method_type'] == 'bart-concept']["human_plan"].tolist()
import operator
concept = list(map(operator.add,list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list1,list2)), list3)), list4)),list5)),list6)),list7)),list8))

list1 = df1.loc[df1['method_type'] == 'gpt-planner']["human_plan"].tolist()
list2 = df2.loc[df1['method_type'] == 'gpt-planner']["human_plan"].tolist()
list3 = df3.loc[df1['method_type'] == 'gpt-planner']["human_plan"].tolist()
list4 = df4.loc[df1['method_type'] == 'gpt-planner']["human_plan"].tolist()
list5 = df1.loc[df1['method_type'] == 'bart-planner']["human_plan"].tolist()
list6 = df2.loc[df1['method_type'] == 'bart-planner']["human_plan"].tolist()
list7 = df3.loc[df1['method_type'] == 'bart-planner']["human_plan"].tolist()
list8 = df4.loc[df1['method_type'] == 'bart-planner']["human_plan"].tolist()
import operator
planner = list(map(operator.add,list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list(map(operator.add, list1,list2)), list3)), list4)),list5)),list6)),list7)),list8))

with open('significance.txt', 'w') as f:
    for list_item in concept:
        f.writelines(str(list_item/8)+'\n')
    for list_item in planner:
        f.writelines(str(list_item/8)+'\n')