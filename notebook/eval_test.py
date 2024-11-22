#!/usr/bin/env python
# coding: utf-8
"""
Created on 2024/11/22 14:15
@author: Zhu Jiang
"""


# %%
from human_eval.data import read_problems
from core import fix_indents, extract_function_body, remove_code_markers
from openai import OpenAI


# %%
problems = read_problems()
print(problems['HumanEval/0']['prompt'])
print(f"{'-'*10}")
print(problems['HumanEval/0']['entry_point'])
print(f"{'-'*10}")
print(problems['HumanEval/0']['canonical_solution'])
print(f"{'-'*10}")
print(problems['HumanEval/0']['test'])

client = OpenAI(
    api_key='sk-fc6c3b63d0bf46b597b5518bdd82bfee',
    base_url='https://api.deepseek.com'
)

model_name = "deepseek-chat"
# model_name = "deepseek-coder"

prompt = problems['HumanEval/0']['prompt']
instruction = (
    "# Please provide only the code implementation for the following function "
    "without any explanations or additional text. Ensure that your code does not "
    "contain more than one consecutive newline character.\n\n"
)
modified_prompt = instruction + prompt


# %%
res = client.chat.completions.create(
    model=model_name,
    messages=[{
        "role": "user",
        "content": modified_prompt,
    }],
    temperature=0.2,
    max_tokens=512,
    top_p=0.95,
    stop=None,
)
output = res.choices[0].message.content
print(output)

# 对生成的代码进行后处理
# 步骤一：去除```标记
output = remove_code_markers(output)
# 步骤二：修正缩进（可选）
output = fix_indents(output)
# 步骤三：提取import和函数定义部分
output = extract_function_body(output)

print(output)
