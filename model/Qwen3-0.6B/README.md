# 在 nano-vLLM 里练习 Qwen3-0.6B FP8

先读懂官方 FP8 checkpoint，用反量化后的 BF16 权重跑通现有模型。接着让 FP8 权重常驻显存，最后练习在 forward 里根据当前输入动态量化激活。这三步可以分别验证加载、存储和计算。

## 0. 准备模型和 BF16 基线

在 WSL 执行：

```bash
cd /home/yhy/yhy_study/my_nano_vllm
conda activate nanovllm
python example.py
hf download Qwen/Qwen3-0.6B-FP8 --local-dir model/Qwen3-0.6B/fp8
```

如果官方站点连接慢，在下载命令前加 `HF_ENDPOINT=https://hf-mirror.com`。已有的 BF16 基线在 `~/huggingface/Qwen3-0.6B`。本练习先用 `tensor_parallel_size=1`、`enforce_eager=True`，固定短 prompt，逐步比较同一组 token 的 logits。看生成结果时先用贪心采样，生成 8 到 16 个 token 即可。

## 1. 先用 safetensors 看 checkpoint

官方模型仍是 dense `qwen3`。读取 `config.json` 的 `quantization_config`，记录量化方法、权重块大小和激活量化方式。官方权重用 128×128 的块缩放。每块有一个 scale。

```bash
python - <<'PY'
import json
from pathlib import Path
from safetensors import safe_open

path = Path("model/Qwen3-0.6B/fp8")
config = json.loads((path / "config.json").read_text())
print(config["model_type"], config["quantization_config"])
for shard in sorted(path.glob("*.safetensors")):
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.layers.0.self_attn.q_proj."):
                t = f.get_tensor(key)
                print(shard.name, key, t.shape, t.dtype)
PY
```

确认 `q_proj.weight` 和对应 scale 的实际名字、dtype、形状。若 scale 键是 `weight_scale_inv`，还要用参考实现核实计算方向。这里把反量化用的乘数记作 `s_dequant`。

## 2. 加载时反量化，复用现有 BF16 Linear

入口在 [model_runner.py](../../nanovllm/engine/model_runner.py)，它已经按 `model_type` 创建模型。把 `quantization_config` 传给 [loader.py](../../nanovllm/utils/loader.py)，用它选择 FP8 checkpoint 处理逻辑。

1. 先做 `dequant_fp8_blockwise(weight, scale, block_size)`。对形状 `[out_features, in_features]` 的权重，scale 网格应为 `[ceil(out_features / 128), ceil(in_features / 128)]`。扩展 scale 时按两个轴各重复 128 次，再裁回原形状。用 FP32 计算 `weight.float() * s_dequant`，最后转 BF16。
2. 用 safetensors 索引（如果存在）或扫描每个分片的键，把 FP8 权重和 scale 配对。分片遍历顺序无关。缺失 scale 或 scale 形状不匹配应直接报错。普通浮点权重继续用原加载路径。
3. 反量化发生在已有 packed mapping 之前。随后复用原来的 `weight_loader`，将 `q_proj/k_proj/v_proj` 写入 `qkv_proj`，将 `gate_proj/up_proj` 写入 `gate_up_proj`。scale 是 loader 的输入，消费后从参数遍历中跳过。
4. 先测一块手算数据，再测一层真实 `q_proj`，最后跑整个模型。用反量化权重直接调用 `F.linear` 作为单层参考，检查输出是否有限并记录误差。

整模型优先用**同一份 FP8 权重的反量化参考**对比 prefill 和连续几步 decode logits。比较时固定每一步输入 token，避免不同采样结果改变后续输入。原始 BF16 checkpoint 可以用来测量化误差，它和官方 FP8 checkpoint 的 logits 允许有差异。此阶段模型中的权重仍是 BF16。

## 3. 保留 FP8 权重，练习量化 BF16 权重

从 [linear.py](../../nanovllm/layers/linear.py) 的一个 Linear 单独练起。把 `weight_fp8` 和对应的 scale 存起来，forward 时临时还原 BF16 并调用 `F.linear`。输出应与第 2 步预先反量化的同一份权重一致。等单层通过再接入整模型，注意不要同时保留该层的完整 BF16 权重副本。

接着用原始 BF16 checkpoint 练习**加载时量化权重**。对每个权重块，在 FP32 下计算：

```text
fp8_max = torch.finfo(torch.float8_e4m3fn).max
s_dequant = absmax(block) / fp8_max
q = clamp(block / s_dequant, -fp8_max, fp8_max).to(float8_e4m3fn)
block_hat = q.float() * s_dequant
```

全零块的 scale 用 1。先做 round-trip，记录权重和单层输出误差。权重加载后固定下来，每次 forward 复用。若要自制兼容 checkpoint，还需要保存 FP8 weight、scale 及对应的量化配置，并用第 2 步从磁盘重新加载验证。

## 4. forward 时动态量化 BF16 激活

每次 forward 进来的激活 `x` 都不同。以 `[tokens, in_features]` 为例，可以先算每个 token 一条 scale，也可以再练习沿输入维按 128 元素一组计算。先用同样的 absmax、缩放、裁剪和 FP8 cast 公式得到 `x_fp8`，再反量化并用 BF16 `F.linear` 验证数值。记得测试全零输入和两组幅值明显不同的激活。

随后选择和 **权重 scale 布局、激活 scale 布局、GPU 和 PyTorch 版本**匹配的 FP8 GEMM 后端。用于 rowwise 的示例不能直接套在官方 128×128 权重 scale 上。先验证单层 GEMM 再跑整模型。对比 BF16、FP8 常驻加临时反量化、原生 FP8 GEMM 三条路径的输出误差、显存和耗时。量耗时时预热，用 CUDA events 或 Nsight Systems，并分别测 prefill 和 decode。临时反量化路径的耗时只能作为功能基线。

## 参考

- [Qwen3-0.6B-FP8 官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B-FP8)
- [官方 FP8 配置](https://huggingface.co/Qwen/Qwen3-0.6B-FP8/blob/main/config.json)
- [TorchAO 量化原理和动态激活示例](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html)

