# 调用模型参数说明

1. temperature=0.2: 
    - 参数控制了生成过程中的随机性。它用于调整输出概率的分布。
    - 数值越高，生成的文本越随机；数值越低，生成的文本越确定。
    - **设为0.2**表示我们希望模型的输出更加保守和一致，减少随机性。
2. max_tokens=512:
   - 参数指定了模型在一次生成中最多可以输出的标记（token）数量。它限制了生成文本的长度，防止生成过长的内容导致资源耗尽或处理困难。
   - 允许模型生成最多512个标记的代码。这个长度通常足够容纳一个完整的函数实现，包括复杂的逻辑和注释。
   - 防止模型因循环生成或未正确终止而生成过长的代码。
3. top_p=0.95:
   - 用于启用核采样(nucleus sampling)策略。 模型会从累积概率达到top_p的词汇集合中进行采样，而不是从整个词汇表中。
   - 这使得生成过程既能保持一定的随机性，又能避免选择罕见或不合适的词。
   - **设为0.95**表示模型在每一步生成时，只考虑累积概率达到95%的候选词。 这样可以过滤掉概率较低的词，减少生成不合理代码的可能性。
4. stop=None:
   - 指定模型在生成文本时的终止条件。当生成的文本中出现指定的停止字符串时，模型会停止生成，返回结果。
   - 未指定任何停止条件，模型将根据自身的结束标记或达到max_tokens限制来停止生成。

调整参数
```python
res = client.chat.completions.create(
    model=model_name,
    messages=[{
        "role": "user",
        "content": prompt,
    }],
    temperature=0.3,  # 略微增加随机性
    max_tokens=600,   # 如果需要更长的代码，可以增加此值
    top_p=0.9,        # 增加多样性
    stop=["\n\n\n"],  # 设置停止条件
)
```
