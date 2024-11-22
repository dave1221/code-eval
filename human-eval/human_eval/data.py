"""
总体功能：
该代码提供了处理JSONL（JSON Lines）格式文件的通用函数，支持读取和写入。
支持处理gzip压缩的JSONL文件，方便处理大型数据集或节省存储空间。

使用场景：
read_problems：读取HumanEval数据集，将任务数据组织为以task_id为键的字典，便于快速查找和访问任务信息。
stream_jsonl：逐行读取JSONL文件，节省内存，适用于大文件的处理。
write_jsonl：将处理后的数据写入JSONL文件，支持追加写入和gzip压缩，方便数据存储和共享。
"""

# 导入需要的模块和类型注解
from typing import Iterable, Dict  # 用于类型提示，Iterable表示可迭代对象，Dict表示字典类型
import gzip                       # 用于处理gzip压缩文件
import json                       # 用于解析和生成JSON格式的数据
import os                         # 提供与操作系统相关的功能，如文件路径操作

# 获取当前脚本所在的目录的绝对路径
ROOT = os.path.dirname(os.path.abspath(__file__))

# 构建HumanEval数据集文件的路径，假设数据集存放在项目的"data"目录下
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")

def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    """
    读取HumanEval数据集，并返回一个以task_id为键的字典。
    参数:
        evalset_file (str): HumanEval数据集文件的路径，默认为HUMAN_EVAL。
    返回:
        Dict[str, Dict]: 一个字典，键为task_id，值为对应的任务信息字典。
    """
    # 使用字典推导式，将stream_jsonl函数生成的每个任务字典，以其"task_id"为键构建新的字典
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    逐行读取JSONL（JSON Lines）格式的文件，并将每一行解析为字典。
    参数:
        filename (str): 要读取的JSONL文件路径，可以是普通文本文件或gzip压缩文件。
    返回:
        Iterable[Dict]: 一个可迭代对象，遍历时会依次返回文件中每一行解析后的字典。
    """
    # 判断文件是否为gzip压缩文件（以".gz"结尾）
    if filename.endswith(".gz"):
        # 以二进制读取模式打开gzip文件
        with open(filename, "rb") as gzfp:
            # 使用gzip打开文件，并以文本模式读取（'rt'表示read text）
            with gzip.open(gzfp, 'rt') as fp:
                # 逐行读取文件
                for line in fp:
                    # 检查行是否为空或只包含空白字符
                    if any(not x.isspace() for x in line):
                        # 将非空行解析为JSON对象，并使用yield返回
                        yield json.loads(line)
    else:
        # 如果不是gzip文件，直接以文本模式打开
        with open(filename, "r") as fp:
            # 逐行读取文件
            for line in fp:
                # 检查行是否为空或只包含空白字符
                if any(not x.isspace() for x in line):
                    # 将非空行解析为JSON对象，并使用yield返回
                    yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    将字典的可迭代对象写入JSONL格式的文件。
    参数:
        filename (str): 要写入的文件路径，可以是普通文本文件或gzip压缩文件。
        data (Iterable[Dict]): 包含字典的可迭代对象，每个字典表示一条记录。
        append (bool): 是否以追加模式写入文件，默认为False（覆盖原文件）。
    功能:
        将data中的每个字典序列化为JSON字符串，并逐行写入到指定文件中。
        如果文件名以".gz"结尾，则以gzip压缩格式写入。
    """
    # 根据append参数，确定文件打开模式
    if append:
        mode = 'ab'  # 以二进制追加模式打开
    else:
        mode = 'wb'  # 以二进制写入模式打开（覆盖原文件）

    # 展开用户目录，例如将"~/path"转换为"/home/user/path"
    filename = os.path.expanduser(filename)

    # 判断文件是否为gzip压缩文件（以".gz"结尾）
    if filename.endswith(".gz"):
        # 以指定模式打开文件
        with open(filename, mode) as fp:
            # 创建gzip文件对象，准备写入压缩数据
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                # 遍历data中的每个字典
                for x in data:
                    # 将字典序列化为JSON字符串，并添加换行符
                    line = json.dumps(x) + "\n"
                    # 将字符串编码为UTF-8字节，并写入gzip文件
                    gzfp.write(line.encode('utf-8'))
    else:
        # 如果不是gzip文件，直接以指定模式打开
        with open(filename, mode) as fp:
            # 遍历data中的每个字典
            for x in data:
                # 将字典序列化为JSON字符串，并添加换行符
                line = json.dumps(x) + "\n"
                # 将字符串编码为UTF-8 字节，并写入文件
                fp.write(line.encode('utf-8'))
