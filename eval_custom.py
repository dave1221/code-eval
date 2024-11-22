#!/usr/bin/env python
# coding: utf-8
"""
Created on 2024/11/22 16:00
@author: Zhu Jiang
"""

from core import fix_indents, run_eval_custom, remove_code_markers, extract_function_body
import os


def generate_batch_completion(
    client, model_name: str, prompt: str, batch_size: int
) -> list[str]:
    """
    批量生成代码完成。
    参数:
    client：模型客户端实例。
    model_name：模型的名称或ID。
    prompt：输入的提示文本。
    batch_size：要生成的样本数量。
    """
    batch_completions = []

    for _ in range(batch_size):
        res = client.chat.completions.create(
            model=model_name,
            messages=[{
                "role": "user",
                "content": prompt,
            }],
            temperature=0.2,
            max_tokens=512,
            top_p=0.95,
            stop=None,
        )
        output = res.choices[0].message.content
        # 对生成的代码进行后处理
        # 步骤一：去除```标记
        output = remove_code_markers(output)
        # 步骤二：修正缩进（可选）
        output = fix_indents(output)
        # 步骤三：提取import和函数定义部分
        output = extract_function_body(output)

        batch_completions.append(output)

    return batch_completions


if __name__ == "__main__":
    num_samples_per_task = 1
    out_path = "results/custom/eval_deepseek.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 初始化模型客户端
    from openai import OpenAI

    client = OpenAI(
        api_key='sk-fc6c3b63d0bf46b597b5518bdd82bfee',
        base_url='https://api.deepseek.com'
    )

    model_name = "deepseek-chat"
    # model_name = "deepseek-coder"

    # 运行评估
    run_eval_custom(
        client=client,
        model_name=model_name,
        num_samples_per_task=num_samples_per_task,
        out_path=out_path,
        generate_batch_completion=generate_batch_completion,
        format_tabs=False,
    )
