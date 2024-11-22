"""
该代码主要用于在HumanEval数据集上评估预训练语言模型的代码生成能力。通过批量生成代码，并对生成的样本进行处理和保存，便于后续的评估和分析。各个函数相互协作，实现了从数据读取、生成、处理到结果保存的完整流程。
可扩展性：您可以根据需要修改num_samples_per_task、format_tabs等参数，或者自定义generate_batch_completion函数以适配不同的模型。
代码规范：代码使用了类型提示和详细的注释，便于理解和维护。
实用性：该脚本可以用于评估不同的模型在代码生成任务上的性能，为模型的改进和优化提供参考。
"""

# 导入需要的函数和类
from human_eval.data import write_jsonl, read_problems  # 用于读取问题和写入JSONL文件
from transformers import (
    PreTrainedModel,       # 预训练模型的基类，用于类型提示
    PreTrainedTokenizer,   # 预训练分词器的基类，用于类型提示
)
from tqdm import tqdm      # 用于显示进度条
import itertools           # 提供高效的迭代器函数
import typing              # 用于类型提示

# 定义一个类型别名，表示批量生成函数的类型
BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]
# BatchGenerator 是一个可调用对象，接受模型、分词器、提示字符串、批量大小，返回字符串列表

# 引用自：https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    """
    过滤生成的代码，只保留第一个函数定义。
    参数:
        completion (str): 模型生成的代码字符串。
    返回:
        str: 过滤后的代码字符串，只包含第一个函数定义。
    """
    # 去除开头的换行符
    completion = completion.lstrip("\n")
    # 按两个连续的换行符分割，返回第一个部分
    return completion.split("\n\n")[0]

def fix_indents(text: str) -> str:
    """
    修正代码缩进，将制表符替换为四个空格。
    参数:
        text (str): 代码字符串。
    返回:
        str: 修正缩进后的代码字符串。
    """
    # 将所有的制表符('\t')替换为四个空格
    return text.replace("\t", "    ")

def split_batch(samples: list[str], size=4) -> list[list[str]]:
    """
    将样本列表分割成指定大小的小批次。
    参数:
        samples (list[str]): 样本字符串列表。
        size (int): 每个小批次的大小，默认为4。
    返回:
        list[list[str]]: 分割后的小批次列表。
    """
    mini_batches = []
    # 通过步长为size的索引遍历样本列表
    for i in range(0, len(samples), size):
        # 切片获取小批次，并添加到mini_batches列表中
        mini_batches.append(samples[i : i + size])
    return mini_batches

def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    format_tabs: bool = False,
):
    """
    运行模型评估，在HumanEval数据集上生成代码并保存结果。
    参数:
        model (PreTrainedModel): 预训练的语言模型。
        tokenizer (PreTrainedTokenizer): 对应的分词器。
        num_samples_per_task (int): 每个任务生成的样本数量。
        out_path (str): 输出结果保存的路径。
        generate_batch_completion (BatchGenerator): 批量生成代码的函数。
        format_tabs (bool): 是否将空格转换为制表符，默认为False。
    """
    # 读取HumanEval数据集中的问题，返回一个以task_id为键的字典
    problems = read_problems()
    # 如果需要只评估部分任务，可以取消以下注释，取前20个任务
    # problems = dict(itertools.islice(problems.items(), 20))
    samples = []  # 用于存储生成的代码样本
    # 初始化进度条，总进度为问题数量乘以每个任务的样本数量
    pbar = tqdm(total=len(problems) * num_samples_per_task)
    # 遍历每个任务的ID
    for task_id in problems:
        # 获取对应的提示(prompt)
        if format_tabs:
            # 如果format_tabs为True，将四个空格替换为制表符
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            # 否则直接使用提示
            prompt = problems[task_id]["prompt"]

        # 使用批量生成函数，生成代码补全
        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task
        )

        # 遍历生成的代码样本
        for sample in batch_completions:
            # 创建结果字典，包含任务ID和生成的代码
            result = dict(
                task_id=task_id,
                completion=sample,
            )
            # 将结果添加到samples列表中
            samples += [result]

        # 更新进度条，增加已完成的样本数量
        pbar.update(num_samples_per_task)

    # 将所有生成的样本写入指定的输出文件，格式为JSON Lines
    write_jsonl(out_path, samples)


def remove_code_markers(output: str) -> str:
    """
    去除输出字符串中的代码块标记，例如```python和```。
    参数:
        output (str): 包含代码块标记的字符串。
    返回:
        str: 去除了代码块标记的代码字符串。
    """
    import re
    # 去除开头的```python或```标记
    output = re.sub(r'^```(?:python)?\s*', '', output.strip())
    # 去除结尾的```标记
    output = re.sub(r'\s*```$', '', output.strip())
    return output.strip()


def extract_function_body(code: str) -> str:
    """
    提取代码中从import语句开始到最后一个return语句之间的部分。
    参数:
        code (str): 代码字符串。
    返回:
        str: 包含import语句和函数定义的代码字符串。
    """
    lines = code.strip().split('\n')
    function_lines = []
    inside_function = False

    for line in lines:
        stripped_line = line.strip()
        # 检查是否为import语句或函数定义的起始行
        if stripped_line.startswith('import ') or stripped_line.startswith('from '):
            function_lines.append(line)
        elif stripped_line.startswith('def '):
            inside_function = True
            function_lines.append(line)
        elif inside_function:
            function_lines.append(line)

    # 截断到最后一个return语句
    last_return_index = None
    for idx, line in enumerate(reversed(function_lines)):
        if 'return' in line:
            last_return_index = len(function_lines) - idx - 1
            break

    if last_return_index is not None:
        function_lines = function_lines[:last_return_index + 1]

    function_code = '\n'.join(function_lines)
    return function_code.strip()



def run_eval_custom(
    client,
    model_name: str,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion,
    format_tabs: bool = False,
):
    problems = read_problems()
    total = len(problems) * num_samples_per_task
    print(f"共需要运行{len(problems)}*{num_samples_per_task}={total}次评估")
    pbar = tqdm(total=total)

    samples = []
    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        instruction = (
            "# Please provide the complete code implementation for the following function, "
            "including any necessary import statements, without any explanations or additional text. "
            "Ensure that your code does not contain more than one consecutive newline character.\n\n"
        )
        modified_prompt = instruction + prompt

        print(f"对{task_id}预测结果, 问题是:\n{prompt}")
        batch_completions = generate_batch_completion(
            client, model_name, modified_prompt, num_samples_per_task
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )
            samples.append(result)

        pbar.update(num_samples_per_task)

    write_jsonl(out_path, samples)