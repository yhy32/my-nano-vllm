# Qwen3-MoE 专家并行阅读指南

目标是沿着一次 forward，看懂权重属于哪张卡、token 怎么找到专家、结果怎么回到下一层 Attention。下面按实现顺序阅读。

## 1. 先确定每张卡持有哪些权重

入口在 `nanovllm/config.py` 和 `nanovllm/engine/model_runner.py`。设置 `enable_expert_parallel=True`，沿用引擎已有的一组进程。例如两卡时 Attention 仍用 TP=2，MoE 专家用 EP=2，每个专家内部用 TP=1。各 rank 都持有完整 router。

看 `nanovllm/models/qwen3_moe.py` 的稀疏层构造。以 4 个专家为例，rank 0 创建专家 0、1，rank 1 创建专家 2、3。模块名保留全局专家 ID。`nanovllm/utils/loader.py` 在读取 tensor 前按名称过滤，远端专家的 weight 和 scale 都跳过。本地专家仍把 gate 和 up 两个完整矩阵依次写入融合参数，down 保存完整矩阵。`nanovllm/layers/linear.py` 让本地专家复用已有的 BF16 / 块 FP8 加载逻辑。

## 2. 给每个 token 选一个发送方

看 `nanovllm/layers/moe.py`。Attention 的输出已在各 rank 上复制。如果两张卡都发送全部 token，同一个专家会重复计算。这里用全局 token 行号对进程数取余决定发送方。输入为 [4, H] 时，rank 0 负责 t0、t2，rank 1 负责 t1、t3。每张卡只对自己的行运行 router。单 token decode 时 rank 1 仍参与通信，它可能持有被选中的专家。

## 3. Dispatch：按目的 rank 打包并交换

假设 top-2 路由如下。每条路线携带一个 H 维向量和一个专家 ID。

| token | 发送方 | 选中专家 | 目的 rank |
|---|---:|---|---|
| t0 | 0 | 0、2 | 0、1 |
| t2 | 0 | 2、3 | 1、1 |
| t1 | 1 | 0、1 | 0、0 |
| t3 | 1 | 2、3 | 1、1 |

rank 0 的发送计数是 [1, 3]，rank 1 的是 [2, 2]。以 rank 1 为例，交换计数后它知道要从 rank 0 接 3 行、从自身接 2 行。这里计数的单位是「token 选中一个专家」的一条路线。举例来说，rank 0 的 t2 同时选中专家 2、3，虽然两位专家都在 rank 1，仍会发两行。

然后用变长 `all_to_all_single` 交换 hidden states 和专家 ID。路由分数和 token 行号留在发送方，供回程使用。

## 4. 本地专家计算与 Combine

接收方按专家 ID 取行，把专家输出写回相同的接收位置。再交换一次专家输出，这次把发送长度和接收长度对调，结果按原来的打包顺序回到发送方。例如 t0 的结果是 score₀ × expert₀(t0) + score₂ × expert₂(t0)。

发送方把加权结果按全局 token 行号累加到 [T, H] 的零张量。此时 rank 0 只有偶数行有效，rank 1 只有奇数行有效。最后一次 `all_reduce(SUM)` 恢复每张卡完整的 [T, H]，接回原来的残差和 TP Attention。这里 all-to-all 负责搬运行，all-reduce 负责逐元素求和。这个教学实现优先让每一步可见，保留了动态分组和计数从 GPU 到 CPU 的同步。

## 5. 运行与验证

在项目根目录、nanovllm 环境中执行：

```bash
torchrun --standalone --nproc-per-node=2 test_moe_ep.py
```

默认用两个 CPU 进程和 Gloo。双 GPU 机器上在命令末尾加 `--cuda`，使用 NCCL。测试使用合成 checkpoint 并和单进程等价的完整专家计算逐元素比较，覆盖短 decode 和不均匀路由。

在原有示例的 `LLM(...)` 处指定 `tensor_parallel_size=2, enable_expert_parallel=True, enforce_eager=True` 即可走整模型入口。像原示例一样用普通 Python 启动，引擎自己创建 worker。专家数需能被进程数整除，Attention 还要满足已有的头数切分要求。

### 对照代码看一行数据

在上表的 rank 0，`indices = [[0, 2], [2, 3]]`，展平就是 [0, 2, 2, 3]。排序后 `token_rows = [0, 0, 1, 1]`，其中本地第 1 行就是全局 t2。回程后这四行仍依次对应 t0、t0、t2、t2。`rank + world * token_rows` 算出全局位置 [0, 0, 2, 2]，`index_add_` 把同一 token 的两份加权结果相加。

### 本地实测范围

两个 CPU 进程通过 Gloo 实际交换了 token。一个 GPU 进程通过 NCCL 跑通了相同入口。两组测试均使用 FP32 和块 FP8 合成专家权重，并与独立逐 token 参考计算比较。覆盖 4 个 token 跨 rank、1 个 token 全发给另一 rank、5 个 token 全发给 rank 0。双 GPU 的 NCCL 传输和完整 30B checkpoint 仍需在多卡机器验证。
