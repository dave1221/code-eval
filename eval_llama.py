"""
该脚本的主要功能是：
加载预训练的LLaMA模型和分词器。
定义一个函数，用于批量生成代码补全，支持生成多个样本。
使用run_eval函数，对模型在特定任务上的性能进行评估，生成结果并保存。
评估过程中，模型将基于提示（可能是代码片段或函数定义），生成代码补全，并对生成的代码进行后处理，然后计算模型在任务上的表现指标。
"""


# 从transformers库中导入需要的模块和类
from transformers import (
    LlamaTokenizer,  # 用于对文本进行分词和编码的分词器
    LlamaForCausalLM,  # LLaMA模型，用于因果语言模型任务
    PreTrainedModel,  # 预训练模型的基类
    PreTrainedTokenizer,  # 预训练分词器的基类
)

# 从自定义的core模块中导入函数
from core import filter_code, run_eval, fix_indents
import os  # 操作系统相关功能，如文件路径操作
import torch  # PyTorch库，用于深度学习模型的构建和训练

# TODO: 使用python-dotenv管理环境变量
# 在这里添加Hugging Face的访问令牌
TOKEN = "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"  # 用于访问受限的模型或数据，需要填写有效的令牌

# 定义一个函数，用于批量生成代码补全
@torch.inference_mode()
def generate_batch_completion(
        model: PreTrainedModel,  # 预训练的语言模型
        tokenizer: PreTrainedTokenizer,  # 对应的分词器
        prompt,  # 输入的提示文本
        batch_size  # 批处理的大小，即生成多少个样本
) -> list[str]:
    # 创建一个包含相同提示的输入列表，长度为batch_size
    input_batch = [prompt for _ in range(batch_size)]
    # 使用分词器对输入批次进行编码，返回PyTorch张量，并将其移动到模型所在的设备（如GPU）
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    # 获取输入序列的长度（即编码后的输入ID的长度）
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    # 使用模型生成新标记（文本），参数如下：
    generated_ids = model.generate(
        **inputs,  # 输入的编码张量
        use_cache=True,  # 启用缓存，以加速生成
        max_new_tokens=512,  # 生成的最大新标记数量
        temperature=0.2,  # 温度参数，控制生成的随机性，越低越保守
        top_p=0.95,  # nucleus sampling，选择概率质量为95%的标记
        do_sample=True,  # 启用采样，否则使用贪心搜索
        eos_token_id=tokenizer.eos_token_id,  # 结束标记的ID
        pad_token_id=tokenizer.eos_token_id,  # 填充标记的ID，使用结束标记代替
    )

    # 解码生成的ID序列，跳过输入部分，只解码新生成的部分
    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],  # 切片，获取新生成的ID部分
        skip_special_tokens=True,  # 跳过特殊标记，如<eos>、<pad>等
    )

    # 对生成的代码进行后处理，如修正缩进和过滤无效代码
    return [filter_code(fix_indents(completion)) for completion in batch_completions]


# 如果脚本作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 设置每个任务要生成的样本数量，例如n=10
    num_samples_per_task = 10
    # 定义输出结果的保存路径
    out_path = "results/llama/eval.jsonl"
    # 确保结果保存的目录存在，不存在则创建
    os.makedirs("results/llama", exist_ok=True)

    # 从预训练模型中加载分词器，这里使用了LLaMA 7B模型
    tokenizer = LlamaTokenizer.from_pretrained(
        "huggyllama/llama-7b",
    )

    # 加载预训练的语言模型，并编译以加速推理
    model = torch.compile(
        LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-7b",  # 指定模型名称
            torch_dtype=torch.bfloat16,  # 设置模型的参数数据类型为bfloat16，节省显存
        )
        .eval()  # 将模型设置为评估模式，关闭训练相关的功能（如dropout）
        .to("cuda")  # 将模型移动到GPU上，以加速计算
    )

    # 调用run_eval函数，对模型进行评估
    run_eval(
        model,  # 预训练的语言模型
        tokenizer,  # 对应的分词器
        num_samples_per_task,  # 每个任务生成的样本数量
        out_path,  # 评估结果的保存路径
        generate_batch_completion,  # 用于生成代码补全的函数
        True,  # 是否在评估过程中显示进度条等详细信息
    )
