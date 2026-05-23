## 深度代码审查结果

### 已验证正确的部分 ✅
- **DPO 损失公式**：数学正确，符合 DPO 论文
- **模型加载流程**：Base → merge SFT LoRA → add DPO LoRA，reference policy 正确冻结
- **Token ID 提取**：`▁Yes`/`▁No` token 处理正确
- **Top-K 策略逻辑**：排序、截取、采样逻辑正确
- **正负样本对**：数据键名匹配正确

---

### Bug 1（严重）— 左填充下 last-token 位置计算错误

[train_llava_dpo.py:135](train/dpo/train_llava_dpo.py#L135)

```python
last_pos = attention_mask.sum(dim=1) - 1  # 错误！
```

设置了 `padding_side="left"`，但用 `sum-1` 取最后 token 位置。左填充时最后一个 token 始终在 `seq_len-1`。**当前 batch_size=1 时不触发**（无 padding），但若调大 batch_size 会导致取错位置。

### Bug 2（显著）— 多卡训练时 score cache 只更新 rank 0

[train_llava_dpo.py:349-426](train/dpo/train_llava_dpo.py#L349)

只有 GPU 0 的 dataset 拿到更新后的 score cache，其他 7 张卡的 dataset 始终用空 cache → 退化为随机采样。**7/8 的训练数据用的是 random 而非 top_k 策略**。

### Bug 3（中等）— falsy 值检查丢弃有效参数

[train_llava_dpo.py:502-510](train/dpo/train_llava_dpo.py#L502)

```python
if args.beta:  # 当 beta=0.0 时为 False，override 被忽略
```

### Bug 4（轻微）— top_k 为 None 时文件名异常

[train_llava_dpo.py:599](train/dpo/train_llava_dpo.py#L599) — 未传 `--top_k` 时生成 `_kNone.json`

### Bug 5（设计）— `limit_train_batches=0.5` 硬编码

[train_llava_dpo.py:720](train/dpo/train_llava_dpo.py#L720) — 静默丢弃 50% 训练数据

---

**Bug 2 是当前训练最大的问题**（top_k 策略实际只在 1/8 数据上生效）。
