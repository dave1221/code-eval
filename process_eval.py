#!/usr/bin/env python
# coding: utf-8
"""
Created on 2024/11/22 14:15
@author: Zhu Jiang
该脚本的主要功能是：
- 读取：从指定的输入文件夹中读取模型生成的代码完成文件，这些文件以JSONL格式存储，每行包含一个任务的task_id和completion。
- 处理：如果指定了--add_prompt参数，对代码完成部分进行清理，包括：
  - 移除Markdown代码块标记，提取纯代码。
  - 移除测试代码、示例用法和主函数入口后的内容，保留核心函数定义。
- 保存：将处理后的数据写入指定的输出文件，供后续评估使用。

假设脚本保存为process_code.py，可以通过以下命令运行：
python process_code.py --path input_folder --out_path output_file.jsonl --add_prompt

--path：替换为您的输入文件夹路径。
--out_path：替换为您希望保存的输出文件路径。
--add_prompt：如果需要处理代码完成部分，请添加此参数。
"""


# 导入需要的模块和函数
from human_eval.data import read_problems, write_jsonl, stream_jsonl  # 从human_eval.data模块导入函数
import glob       # 用于文件路径模式匹配
from tqdm import tqdm  # 用于显示进度条
import argparse   # 用于解析命令行参数

# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument("--path", type=str, help="输入文件夹的路径，包含待处理的JSONL文件")  # 输入文件夹路径参数
parser.add_argument("--out_path", type=str, help="输出文件的路径")  # 输出文件路径参数
parser.add_argument("--add_prompt", action="store_true", help="是否添加提示处理代码")  # 是否添加提示的标志

# 解析命令行参数
args = parser.parse_args()

# 使用glob获取输入路径下的所有JSONL文件，并进行排序
files = sorted(glob.glob(args.path + "/*.jsonl"))
print("{} files in {}".format(len(files), args.path))  # 打印找到的文件数量和路径

# 读取HumanEval数据集中的问题，返回一个以task_id为键的字典
problems = read_problems()

# 初始化输出列表和计数器
output = []  # 用于存储处理后的代码
a = 0        # 计数器，用于统计在处理过程中出现异常的次数

# 遍历每个代码文件，使用tqdm显示进度条
for code_file in tqdm(files, total=len(files)):
    # 使用stream_jsonl读取JSONL文件中的所有代码
    codes = [c for c in stream_jsonl(code_file)]
    # 如果指定了--add_prompt参数，则进行代码处理
    if args.add_prompt:
        for code in codes:
            # 获取task_id和对应的提示
            task_id = code["task_id"]
            prompt = problems[task_id]["prompt"]
            # 获取代码完成（模型生成的代码）
            completion = code["completion"]
            # 移除回车符号，统一换行符
            completion = completion.replace("\r", "")
            # 处理包含Markdown代码块的情况
            if "```python" in completion:
                # 找到"```python"的位置
                def_line = completion.index("```python")
                # 截取从"```python"开始的字符串
                completion = completion[def_line:].strip()
                # 移除"```python"标记
                completion = completion.replace("```python", "")
                # 尝试找到代码块的结束标记"```"
                try:
                    next_line = completion.index("```")
                    # 截取代码块内容
                    completion = completion[:next_line].strip()
                except:
                    # 如果未找到结束标记，计数器加1，并打印未处理的代码
                    a += 1
                    print(completion)
                    print("================\n")
            # 处理包含主函数入口的情况，移除后续内容
            if '__name__ == "__main__"' in completion:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()
            # 处理包含示例用法的情况，移除后续内容
            if "# Example usage" in completion:
                next_line = completion.index("# Example usage")
                completion = completion[:next_line].strip()
            # 更新代码完成部分
            code["completion"] = completion
    # 将（处理后的）代码添加到输出列表中
    output += codes

# 保存处理后的代码到指定的输出文件
print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)
# 打印在处理过程中未找到代码块结束标记的次数
print(a)
