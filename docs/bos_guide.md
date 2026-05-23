# BOS 模型上传指南

> 日期：2026-04-22
> 工具：bcecmd v0.5.10
> BOS Bucket：nlp-data-app-models
> 操作路径：

---

## 一、bcecmd 工具准备

容器内已内置 bcecmd，路径：

```
/root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10/bcecmd
```

加入 PATH（每次新 shell 需执行）：

```bash
chmod +x /root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10/bcecmd
export PATH=$PATH:/root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10
```

---

## 二、鉴权配置

### 2.1 凭证来源

在**有权限的源机器**上执行以下命令，拿到 AK/SK：

```bash
cat /root/.go-bcecli/credentials
```

输出的鉴权信息如下：

```ini
[Defaults]
Ak = <YOUR_AK>
Sk = <YOUR_SK>
Sts = 
```

### 2.2 清除旧配置（重要！）

系统可能残留两处旧的 bcecmd 鉴权文件，**必须先清除**，否则会干扰新配置导致 `Access Denied`：

```bash
rm -f /root/.go-bcecli/credentials
rm -f /root/.bcecmd/config
```

> **背景：** `/root/.go-bcecli/credentials` 是 go-bcecli 工具的凭证文件，bcecmd 在某些情况下也会读取它，若其中 AK/SK 与目标 bucket 无权限会导致 `Access Denied`。

### 2.3 写入配置

> ⚠️ **必须使用 `bcecmd --configure` 方式写入，不可直接手写 config 文件。**
> `bcecmd --configure` 除写入配置外还会做内部初始化，直接写文件会跳过该步骤导致 `Access Denied`。

将 `YOUR_AK` / `YOUR_SK` 替换为实际值后一键执行：

```bash
export PATH=$PATH:/root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10

printf "YOUR_AK\nYOUR_SK\n\nbj\nbj.bcebos.com\nyes\n\n7\nyes\n10\n10\n10\n12\nno\n" | \
  bcecmd --configure
```

各字段含义（按 printf 顺序对应交互提示）：

| 顺序 | 提示内容 | 填写值 |
|---|---|---|
| 1 | Access Key ID | 从源机器获取的 AK |
| 2 | Secret Access Key ID | 从源机器获取的 SK |
| 3 | Security Token | 留空（直接回车） |
| 4 | Default Region Name | `bj` |
| 5 | Default Domain | `bj.bcebos.com` |
| 6~13 | 其余选项 | 全部回车使用默认值 |

### 2.4 验证鉴权

```bash
bcecmd bos ls bos:/nlp-data-app-models/
# 能列出目录内容即为成功
```

---

## 三、BOS 目录结构约定

```
bos:/nlp-data-app-models/
└── open_source_models/
    ├── GLM/                   ← GLM 系列模型统一放这里
    │   └── GLM_decay_0407_hf/
    ├── Qwen2.5-7B-Instruct/
    ├── DeepSeek-V3-Base/
    └── ...
```

**注意：** GLM 系列模型统一放在 `open_source_models/GLM/` 子目录下，便于管理。

---

## 四、模型上传命令

### 4.1 单次上传

```bash
export PATH=$PATH:/root/paddlejob/workspace/env_run/linux-bcecmd-0.5.10
export MODEL="GLM_decay_0407_hf"
LOG="/root/paddlejob/workspace/env_run/logs/bos_upload_${MODEL}.log"

cd /root/paddlejob/workspace/env_run
nohup bcecmd bos cp -r ./models/$MODEL/ bos:/nlp-data-app-models/open_source_models/GLM/$MODEL/ \
  > "$LOG" 2>&1 &

echo "Upload started, PID: $!, log: $LOG"
```

### 4.2 监控上传进度

```bash
# 实时查看日志
tail -f /root/paddlejob/workspace/env_run/logs/bos_upload_${MODEL}.log

# 确认进程存活
ps aux | grep bcecmd | grep -v grep
```

### 4.3 上传完成校验

```bash
# 列出 BOS 目标目录，确认文件数量
bcecmd bos ls bos:/nlp-data-app-models/open_source_models/GLM/GLM_decay_0407_hf/ | wc -l

# 对比本地文件数
ls /root/paddlejob/workspace/env_run/models/GLM_decay_0407_hf/ | wc -l
```

---

## 五、常见问题

### Q1: `Access Denied` 错误

**根本原因按顺序排查：**

1. **未使用 `bcecmd --configure` 写入配置**：直接手写 `~/.bcecmd/config` 文件会跳过内部初始化，导致鉴权失败。必须使用第 2.3 节的 `printf | bcecmd --configure` 方式。

2. **残留旧配置干扰**：系统存在 `/root/.go-bcecli/credentials` 旧文件，bcecmd 会读取并优先使用。先按第 2.2 节清除，再重新执行第 2.3 节。

3. **AK/SK 对该 bucket 无权限**：确认凭证来自正确的源机器（参考第 2.1 节）。

### Q2: 上传中断后如何续传

bcecmd 支持断点续传，直接重新执行相同的 `cp -r` 命令即可，已上传的分片会自动跳过：

```bash
bcecmd bos cp -r ./models/$MODEL/ bos:/nlp-data-app-models/open_source_models/GLM/$MODEL/
```

### Q3: 新机器快速复用

1. 从有权限的源机器获取 AK/SK：`cat /root/.go-bcecli/credentials`
2. 清除旧配置（第 2.2 节）
3. 用 `printf | bcecmd --configure` 写入配置（第 2.3 节）
4. 执行 `bcecmd bos ls` 验证（第 2.4 节）
5. 执行上传命令（第四节）

---

## 六、本次上传记录

| 字段 | 值 |
|---|---|
| 模型名 | GLM_decay_0407_hf |
| 本地路径 | `/root/paddlejob/workspace/env_run/models/GLM_decay_0407_hf/` |
| 模型大小 | 200G（43 个 safetensors） |
| BOS 目标路径 | `bos:/nlp-data-app-models/open_source_models/GLM/GLM_decay_0407_hf/` |
| 上传日志 | `/root/paddlejob/workspace/env_run/logs/bos_upload_GLM_decay_0407_hf.log` |
| 上传时间 | 2026-04-22 |
