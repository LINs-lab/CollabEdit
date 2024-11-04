import argparse
import json
from numpy import mean
from pprint import pprint
import os
import torch
import torch.nn.functional as F


gpt_j_res_dir = "./results/EleutherAI_gpt-j-6B/round_results/MEMIT_easy_multi.json"

gpt2_xl_res_dir = "./results/gpt2-xl/round_results/MEMIT_easy_multi.json"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["GPT-J", "GPT2-XL"],
    default="GPT2-XL",
    required=True,
)
parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
args = parser.parse_args()
# result_dir = gpt2_xl_res_dir


def levenshtein_distance(str1, str2):
    # 处理空字符串的情况
    if not str1:
        return len(str2)
    if not str2:
        return len(str1)

    # 初始化矩阵
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # 初始化边界条件
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # 填充矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1],
                               dp[i - 1][j - 1]) + 1

    return dp[len1][len2]


def are_strings_similar(str1, str2, max_diff):
    distance = levenshtein_distance(str1, str2)
    return distance <= max_diff


max_diff = 500

if args.model == "GPT-J":
    result_dir = gpt_j_res_dir
else:
    result_dir = gpt2_xl_res_dir

with open(result_dir, "r") as f:
    all_results = json.load(f)
avg_before = 0
avg_after = 0
succ_before = 0
succ_after = 0
nums = 1000
for record in all_results:
    new_object = record["edit"]["new_object"]
    conflict_obj = record["edit"]["conflict_object"]["label"]
    target_obj = record["edit"]["object"]["label"]
    # if are_strings_similar(new_object, conflict_obj, max_diff) or are_strings_similar(target_obj, conflict_obj, max_diff):
    #     nums = nums - 1
    #     continue
    edit_after = record["edit-after"]
    edit_before = record["edit-before"]
    delta_before = edit_before["target_conflict"]["rewrite"] - \
        edit_before["target_new"]["rewrite"]
    avg_before = avg_before + delta_before
    if delta_before > 0:
        succ_before = succ_before + 1

    delta_after = edit_after["target_conflict"]["rewrite"] - \
        edit_after["target_new"]["rewrite"]
    # print(edit_after["target_conflict"]["rewrite"])
    avg_after = avg_after + delta_after
    if delta_after > 0:
        succ_after = succ_after + 1


avg_after = avg_after / nums
avg_before = avg_before / nums
succ_after = succ_after / nums
succ_before = succ_before / nums

# avg_after = avg_after
# avg_before = avg_before
# succ_after = succ_after
# succ_before = succ_before
print("nums: ", nums)
print("avg_after: ", avg_after)
print("avg_before: ", avg_before)
print("succ_after: ", succ_after)
print("succ_before: ", succ_before)
