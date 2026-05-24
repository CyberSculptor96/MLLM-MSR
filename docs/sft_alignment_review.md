## SFT 训练脚本对齐与正确性审查

审查对象：
- `train/dpo/train_qwen_sft.py`
- `train/dpo/train_llava_sft.py`

---

### 总结结论

- **整体训练框架已基本对齐**：两份脚本都遵循 `加载模型 → 注入 LoRA → 读取 SFTDataset → DataLoader → Lightning Trainer → 每 epoch 保存 LoRA` 的主流程。
- **核心监督逻辑尚未完全对齐**：`Qwen` 版本只监督 assistant answer，`LLaVA` 版本当前会对整段 prompt + answer 一起计算 loss。
- **正确性上 Qwen 实现更标准**：`train/dpo/train_qwen_sft.py` 的 label masking 更符合 instruction tuning / answer-only SFT 的常见做法。
- **LLaVA 存在一个重要设计偏差**：`train/dpo/train_llava_sft.py` 没有屏蔽用户 prompt 区域，训练信号会被 prompt token 稀释。

---

### 已对齐的核心部分 ✅

- **任务目标一致**：两者都训练多模态 `Yes/No` 推荐器，数据都来自 `sft_train.json`。
  - `train/dpo/train_qwen_sft.py:214-219`
  - `train/dpo/train_llava_sft.py:201-206`

- **Prompt 语义一致**：都使用相同的推荐判断模板。
  - `train/dpo/train_qwen_sft.py:44-50`
  - `train/dpo/train_llava_sft.py:46-52`

- **LoRA 训练主流程一致**：
  - 加载 base model
  - 自动查找线性层作为 target modules
  - 注入 LoRA
  - 仅训练可训练参数
  - `train/dpo/train_qwen_sft.py:237-264`
  - `train/dpo/train_llava_sft.py:224-248`

- **训练框架一致**：都用 Lightning + DeepSpeed Stage 2，多卡训练、梯度累积、梯度裁剪、每 epoch 保存 adapter。
  - `train/dpo/train_qwen_sft.py:286-305`
  - `train/dpo/train_llava_sft.py:271-290`

---

### 关键差异 1（最重要）— Label Masking 没有对齐

#### Qwen：只监督 assistant answer

`train/dpo/train_qwen_sft.py:158-184`

Qwen 版本的处理逻辑是：
- 先构造完整 chat message：user + assistant
- 再通过 tokenizer 找到 assistant 起始位置
- 将 assistant 响应之前的 token 全部置为 `-100`
- 最终只对 assistant answer 的 token 计算 loss

这意味着它优化的是：
- 给定 `用户历史 + item 图文输入`
- 让模型学会输出 `Yes` / `No`

这是标准的 answer-only SFT。

#### LLaVA：当前监督的是整段 prompt + answer

`train/dpo/train_llava_sft.py:151-166`

LLaVA 版本当前做法：
- 文本直接拼成：
  - `"[INST] <image>\n{prompt} [/INST] {ground_truth}"`
- `labels = input_ids.clone()`
- 只把 padding token 设为 `-100`

也就是说，除了 padding 以外：
- prompt token 被监督
- `[/INST]` 之前的 instruction token 被监督
- 最后的 `Yes/No` answer 也被监督

这与 Qwen 的 answer-only 逻辑**没有对齐**。

#### 影响

这会带来两个问题：
- **训练目标不纯**：模型不仅在学会回答 `Yes/No`，还在学复现 prompt 模板。
- **有效监督被稀释**：真正重要的 answer token 很少，而 prompt token 很长，loss 的主要贡献可能来自 prompt 区域。

对于当前任务，更合理的目标应该是：

- 只优化 `P(Yes|prompt,image)` / `P(No|prompt,image)`
- 不把用户输入 prompt 本身作为学习目标

**结论**：这一点是两份脚本当前最核心的不对齐处，也是 `train_llava_sft.py` 的主要正确性问题。

---

### 关键差异 2 — 输入模板形式不同，但语义对齐

#### Qwen 使用 chat template

`train/dpo/train_qwen_sft.py:126-145`

Qwen 通过：
- `messages = [{role: user, ...}, {role: assistant, ...}]`
- `processor.apply_chat_template(...)`

来构造输入。

#### LLaVA 使用 `[INST] ... [/INST]`

`train/dpo/train_llava_sft.py:151-153`

LLaVA 通过直接拼接 instruction template：
- `"[INST] <image>\n{prompt} [/INST] {ground_truth}"`

#### 判断

这两种方式虽然模板协议不同：
- Qwen：chat markup
- LLaVA：instruction markup

但**语义层面是对齐的**：
- 都是在给模型一个多模态用户输入
- 都要求模型输出 `Yes/No`

因此，**模板形式不同不是问题，监督边界不同才是问题**。

---

### 关键差异 3 — Batch 输出接口不同，但功能等价

#### Qwen

`train/dpo/train_qwen_sft.py:76-82`

Qwen 的 `collate_fn` 返回 dict，`training_step()` 直接：

```python
outputs = self.model(**batch)
```

#### LLaVA

`train/dpo/train_llava_sft.py:97-108`

LLaVA 的 `collate_fn` 返回 tuple，`training_step()` 手工拆包：

```python
input_ids, attention_mask, pixel_values, image_sizes, labels = batch
```

#### 判断

这是接口风格差异，不是逻辑错误。

---

### 正确性检查结果

#### `train/dpo/train_qwen_sft.py` ✅ 基本正确

- 语法通过。
- answer-only masking 思路正确。
- `processor.apply_chat_template()` 与 Qwen2.5-VL 模型形态匹配。
- `training_step()` / optimizer / trainer 配置自洽。

#### 需要注意的点

- `assistant_start = ... + 1` 依赖固定的 assistant header 后存在一个换行 token：
  - `train/dpo/train_qwen_sft.py:180`
- 这在当前模板下大概率可用，但实现略脆弱；如果 tokenizer/chat template 细节变化，mask 边界可能错位。

这更像**稳健性问题**，不是当前明确 bug。

---

#### `train/dpo/train_llava_sft.py` ⚠️ 存在重要偏差

- 语法通过。
- 模型能训练起来，但监督目标没有严格聚焦 answer token。
- `labels` 只 mask padding：
  - `train/dpo/train_llava_sft.py:163-166`
- 这会导致训练目标与 Qwen 版不一致，也偏离更标准的 SFT 设计。

#### 额外注意点

- `prepare_model_for_kbit_training(model)` 被调用：
  - `train/dpo/train_llava_sft.py:246`
- 但模型并未以 4bit / 8bit 量化形式加载。
- 这未必直接出错，但语义上不够干净，容易让后续读代码的人误以为这里使用了 QLoRA / k-bit 训练链路。

---

### 最终判断

如果问题是：

> 两份脚本的核心代码逻辑是否已经对齐？

答案是：

- **训练管线层面：基本对齐**
- **核心监督逻辑层面：尚未完全对齐**

如果问题是：

> 两份脚本是否都正确？

答案是：

- `train_qwen_sft.py`：**总体正确，且更接近标准实现**
- `train_llava_sft.py`：**可以运行，但在 label masking 上存在重要设计偏差，应视为需要修正**

---

### 建议的修正方向

优先建议修正 `train/dpo/train_llava_sft.py` 的 `sft_collate_fn()`：

- 保留当前 prompt 构造方式不变
- 但在构造 `labels` 时：
  - 屏蔽 `[INST] ... [/INST]` 全部 token
  - 只保留答案 `Yes` / `No` 对应 token 的监督

这样可以让它和 `train_qwen_sft.py` 的核心监督逻辑真正对齐，也更符合后续 DPO 训练对 `P(Yes)` / `P(No)` 的使用方式。
