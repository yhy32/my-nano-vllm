# nano-vLLM 后续路线图

## 当前基线

- Qwen3-0.6B FP8 已在单卡 eager 模式跑通。112 个 FP8 Linear 通过自定义 CUDA 算子按 128×128 块反量化，再用 BF16 `F.linear` 计算。
- MoE 已有 Router、Top-K、Dispatch 和 Combine 的单卡路径。先用小尺寸随机 checkpoint 做正确性验证，再用完整 Qwen3-30B-A3B 验证真实负载。

## 第一阶段：融合 FP8 权重反量化和矩阵乘法

**目标**是在一个算子里读取 FP8 权重及 scale，按 tile 反量化并累加，直接输出投影结果。当前 QKV 的完整 BF16 临时权重约 8 MiB，融合后可以省去这块权重的写出和重新读取。

- [ ] 先做单 token decode 的 QKV 投影。用 `X[1, 1024]`、`W_fp8[4096, 1024]` 和 `scale_inv[32, 8]` 验证 K 维跨越 128 元素时能正确切换 scale。
- [ ] 把逐 tile 反量化接到多 token prefill 的矩阵乘法，覆盖不同 token 数，再扩展到其他 FP8 Linear。
- [ ] 保留「独立 CUDA 反量化 + BF16 Linear」作为参照。对比输出误差、整个投影的 GPU 耗时和临时显存，分别记录 decode 和 prefill。

## 第二阶段：支持 MoE Expert Parallelism（EP）

**目标**是把专家分布在多张 GPU 上，将 token 发送给它们各自选中的专家，并把结果返回原 rank 完成加权合并。

- [ ] 先在两张 GPU 上用小尺寸 MoE 定义 expert ID 到 rank 的映射。用 BF16 跑通，复用现有 Router 和 Top-K。
- [ ] 实现按目标 rank 分组、NCCL All-to-All dispatch、本地专家计算和结果回传。保留 token ID 和 top-k 槽位，最后按路由权重 combine。
- [ ] 对照单卡结果，覆盖 token 发往多个 rank、某个 rank 没收到 token、专家负载不均等情况。两卡通过后再跑完整 Qwen3-30B-A3B，分别测 dispatch、专家计算和回传耗时。

## 第三阶段：支持 Prefill/Decode（PD）分离

**目标**是让不同 worker 分别执行 prefill 和 decode，通过交接 KV cache 接续同一个请求。

- [ ] 先做单机双 worker 和单请求。明确 prefill 交付第一枚生成 token、请求位置、采样参数和有效 KV 数据的时机；decode 收到后分配本地 block、建立本地 block table，再继续生成。
- [ ] 固定逐步输入 token，对比单体引擎与分离引擎的 decode logits。验证跨 KV block 的交接，以及请求结束或取消后两端的缓存释放。
- [ ] 单请求通过后再加并发。分别测 TTFT、KV 交接耗时和后续每 token 延迟，并记录端到端吞吐。

