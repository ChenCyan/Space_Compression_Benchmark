# 多波段图像压缩方法 Benchmark 实验报告

> **数据集**: Berlin-Urban-Gradient（HyMap 传感器, 111 波段, 80×80, 14-bit 有效位深）
> **测试集**: 16 patches
> **测试平台**: 2× Tesla V100 GPU / Intel Xeon CPU, Docker (torch 2.5.1+cu121, glymur 0.13.8, numba 0.65)
> **CSV数据**: `results/benchmark_all.csv`（44 个 codec 配置）

---

## 1. 方法总览

| 类别 | 方法 | 类型 | 参数数量 | 运行设备 |
|---|---|---|---|---|
| 经典—无损 | CCSDS-123.0-B-1 | 预测+熵编码 | 0（算法型） | CPU |
| 经典—无损/有损 | JPEG2000 | 小波变换+位面编码 | 0（算法型） | CPU |
| 经典—近无损 | KLT+DWT | PCA+KLT + JPEG2000 | 0（算法型） | CPU |
| 学习型 | CAE1D / CAE3D | 卷积自编码器 | ~0.1–1M | GPU |
| 学习型 | SSCNet | 光谱信号压缩网络 | ~0.5M | GPU |
| 学习型 | HYCOT | Transformer+卷积 混合 | ~1M | GPU |
| 学习型 | **HyCASS** | Swin Transformer+CR Adapter | ~2M | GPU |

---

## 2. 各方法原理详解

### 2.1 CCSDS-123.0-B-1（无损预测编码）

**原理**：基于邻域像素的线性预测。对每个像素，用已编码的空间邻域（上、左、左上）和光谱邻域（前几个波段同位置）计算局部差值向量，与自适应权重向量做内积得到预测值。预测残差经 GPO2（Golomb-Power-of-2）熵编码输出比特流。

```
s(x,y,z) → local_sum → local_diff_vector → weight·diff → ŝ → δ = s−ŝ → mapper → GPO2 encode → bitstream
                                                                      ↑
                                                            weight update（误差反馈）
```

**CR 控制**：**不可调**——无损模式下 CR 由数据熵决定（Berlin 数据 CR≈1.2）。近无损（B-2）模式通过设置 `PAE_bound` 控制——允许量化步长=(2M+1)，M 越大 CR 越高但误差越大。**当前 B-2 待实现。**

**优点**：低复杂度（仅加法和移位），适合星载硬件；CCSDS 国际标准；有近无损模式。
**缺点**：纯 CPU 单线程实现吞吐极低（1.5 MB/s）；CR 低（无损仅 1.2×）；逐像素串行依赖难以并行。

---

### 2.2 JPEG2000（小波变换编码）

**原理**：对图像做离散小波变换（DWT）分解为多尺度子带，然后对子带系数做位面编码（EBCOT）+MQ 算术编码。无损模式用 5/3 可逆整数小波，有损模式用 9/7 不可逆浮点小波。多分量模式（MCT）对光谱维做可逆颜色变换进一步去相关。

```
(C,H,W) → MCT光谱变换 → 2D-DWT逐分量 → 位面截断+EBCOT → JP2码流
```

**CR 控制**：`cratio` 参数直接指定目标压缩比。编码器据此截断低位面，使码流大小≈原始/cratio。

**优点**：ISO 国际标准，成熟稳定；一份码流可渐进解码；多分量 MCT 光谱去相关效果好（Berlin 上 lossless CR=2.3×）。
**缺点**：通过 glymur 调 OpenJPEG C 库，临时文件 I/O + 逐分量编码开销大；低 CR 时有损模式质量退化较快。

---

### 2.3 KLT+DWT（PCA 光谱降维 + JPEG2000 空间编码）

**原理**：先用 PCA（主成分分析）将 111 个原始光谱波段投影到更低维的 `nc` 个主成分空间（KLT = Karhunen-Loève Transform），然后对每个主成分分量用 JPEG2000 做空间编码。

```
(111,80,80) → PCA → (nc,80,80) → JPEG2000 逐分量 → 码流
                ↓
         basis(nc×111) + mean(111) → 侧信息
```

**CR 控制**：双旋钮。`nc` 控制光谱压缩倍数（111→nc），`cratio` 控制每个分量的空间压缩倍数。总 CR ≈ (111/nc) × cratio × 基础系数。侧信息（basis+mean）计入 CR 计算。

**优点**：光谱降维+空间编码分离，灵活可调；不依赖 GPU；同 CR 下质量远优于纯 JPEG2000。
**缺点**：每样本单独 fit PCA（真实场景应用需用训练集基）；nc<56 时 OpenJPEG 不支持多分量编码，退回逐分量模式。

---

### 2.4 CAE1D / CAE3D（卷积自编码器）

**原理**：CAE1D 用 1D 卷积沿光谱维做压缩（conv1d 跨波段），空间维不做压缩。CAE3D 用 3D 卷积同时压缩光谱和空间维。两者都是 encoder→bottleneck→decoder 的标准自编码器架构。

```
CAE1D:  (C,H,W) → Conv1D_enc(C→L) → Latent(L,H,W) → Conv1D_dec(L→C) → (C,H,W)
CAE3D:  (C,H,W) → Conv3D_enc(C→L, H/2, W/2) → Latent → Conv3D_dec → (C,H,W)
```

**CR 控制**：固定 bottleneck 通道数 L。每个 CR 值对应一个独立训练的模型（cr004→L=28, cr008→L=14, cr016→L=7, cr032→L=4, cr051→L=2）。

**CAE1D 优点**：极简单（仅 1D conv），吞吐高（60 MB/s）；在低 CR 下 PSNR 与 HyCASS 接近（CR≈4 时 52.1 vs 49.5 dB）。
**CAE1D 缺点**：纯光谱压缩，空间信息完全保留（bpppc 高）。
**CAE3D 缺点**：质量很差（CR=4 时 PSNR 仅 31.7 dB），3D 卷积对 80×80 空间维的有效信息捕获不足。

---

### 2.5 SSCNet（Spectral Signals Compressor Network）

**原理**：将每个像素的光谱向量作为独立信号，用全连接层做光谱压缩。纯光谱编码，无空间操作。

**CR 控制**：同 CAE — 每个 CR 值一个独立训练模型（cr004–cr1024）。

**优点**：架构最简单（纯 MLP），吞吐极高（最高 170 MB/s），超轻量（~0.5M 参数）。
**缺点**：**质量极差**——CR=4 时 PSNR 仅 25.8 dB（比 JPEG2000 低 45 dB）。从 cr=4 到 cr=1024（562×），PSNR 从 26.1 降到 24.1 — 几乎不随 CR 变化，说明瓶颈信息早已饱和。这个架构基本不适合作为高光谱图像压缩的实用方案。

---

### 2.6 HYCOT（HyCASS Transformer 变体）

**原理**：HyCASS 的前身/变体，同样使用 Transformer 块，但缺少 CR Adapter 模块和空间降采样阶段。编码器和解码器直接连接。

**CR 控制**：同 CAE — 每个 CR 一个独立模型（cr004–cr128）。

**优点**：中低 CR 下质量合理（CR≈2 时 PSNR=46.5 dB）；吞吐稳定（~35 MB/s）。
**缺点**：训练不稳定——cr=16 的 PSNR（41.9）反而低于 cr=32（44.0），非单调 RD 曲线；Transformer 自注意力在 80×80 小分辨率下效率不高。

---

### 2.7 HyCASS（可调时空谱压缩网络）⭐

**原理**：论文主体方法（Fuchs et al., JSTARS 2025）。核心创新是 **CR Adapter 模块**——一个 1×1 卷积将特征通道数压缩到 `L`（bottleneck），使**同一个架构**可以覆盖不同 CR。包含六个模块：
1. Spectral Encoder（1×1 Conv, 光谱初始投影）
2. Spatial Encoder（Swin Transformer + 2× 下采样, 空间压缩）
3. CR Adapter Encoder（1×1 Conv C→L, 瓶颈压缩）
4. CR Adapter Decoder（1×1 Conv L→C, 瓶颈解压）
5. Spatial Decoder（Swin Transformer + 2× 上采样）
6. Spectral Decoder（1×1 Conv, 光谱重建）

```
(C,H,W) → 光谱Enc → 空间Enc(↓2×) → CR适配(L) → 空间Dec(↑2×) → 光谱Dec → (C,H,W)
```

**CR 控制**：两个维度——**① CR Adapter 瓶颈 L**（光谱压缩，spatial0x 的 CR=4–50）和 **② 空间降采样 stages**（spatial2x 的 CR=222–1776，通过 2 级 2× 下采样实现 16× 空间压缩 + L 通道瓶颈）。每个 (L, stages) 组合对应一个独立训练的权重。

**优点**：
- 低 CR 区质量良好（CR=4 时 PSNR=49.5 dB，仅次于 CAE1D 和 JPEG2000）
- 高 CR 区唯一可用的方法（CR>100 时仅 HyCASS 和 SSCNet 可达到，但 SSCNet 质量极差）
- CR 范围最广（2.1×–822×），是所有学习型方法中唯一可同时覆盖低 CR 和高 CR 的
- throughput 高（GPU 上 78–190 MB/s）

**缺点**：
- PAE 大（990–5660 DN），没有误差边界保证
- CR=4 时质量低于 JPEG2000 MCT（49.5 vs 70.8 dB，差 21 dB）— 学习型在低 CR 区不敌经典方法
- 高 CR 时信息瓶颈饱和（cr=822 PSNR=24.5 vs cr=427 PSNR=26.3，仅改善 1.8 dB）
- Swin Transformer 的 window_size 对 80×80 小分辨率不匹配（需 4–8 的窗）

---

## 3. 实验结果

### 3.1 压缩比 vs 吞吐量（全方法，含无损）

![CR vs Throughput](results/plots/cr_vs_throughput.pdf)

**关键发现**：
- **GPU vs CPU 鸿沟**：学习型方法（GPU）吞吐 10–190 MB/s，经典方法（CPU）仅 1.5–7.4 MB/s。两者差 1–2 个数量级。
- **CCSDS-123 是唯一纯 CPU 边界**：1.5 MB/s，远低于 JPEG2000（~5 MB/s）和 KLT+DWT（~7 MB/s）。
- **SSCNet/HyCASS 吞吐最高**：小模型+大 batch→GPU 利用率高。但 SSCNet 质量差，HyCASS 是唯一"高吞吐+质量可接受"的选择。
- **JPEG2000 无损 vs 有损吞吐几乎相同**（4.9 vs 3.9–4.6 MB/s）：说明 JPEG2000 的速率控制不增加计算开销。

---

### 3.2 率失真曲线：CR vs PSNR（有损/近无损方法）

![CR vs PSNR](results/plots/cr_vs_psnr.pdf)

**关键发现**：

| CR 区间 | 最优方法 | PSNR | 说明 |
|---|---|---|---|
| CR≈4 | **JPEG2000 MCT** | 70.8 dB | 经典方法碾压学习型 |
| CR≈4 | KLT+DWT nc=28 | 64.4 dB | 光谱降维有代价 |
| CR≈8 | JPEG2000 | 58.4 dB | 仍领先 |
| CR≈16 | JPEG2000 / CAE1D | 48.9 / 47.6 dB | 学习型开始追平 |
| CR≈28 | **HyCASS cr=50** | 39.1 dB | 学习型超越 KLT+DWT（35.3 dB） |
| CR>100 | **HyCASS（唯一）** | 24.5–31.2 dB | 经典方法无法达到此 CR |

**趋势**：
- 在低 CR 区（≤8），经典方法（JPEG2000, KLT+DWT）全面优于学习型，差距可达 20+ dB。
- CR>16 后，学习型方法（HyCASS, CAE1D）开始反超。
- CAE3D、SSCNet 在所有 CR 下质量都差，架构不适合该任务。
- HyCASS 在 CR=2–822 范围内 PSNR 从 49.6→24.5 dB，RD 曲线平滑单调。

---

### 3.3 率失真曲线：CR vs PAE（有损/近无损方法）

![CR vs PAE](results/plots/cr_vs_pae.pdf)

**关键发现**：
- **JPEG2000 PAE 控制最好**：CR=4 时 PAE 仅 16 DN（PSNR=70.8 dB），CR=16 时 377 DN — 误差分布均匀。
- **所有学习型方法的 PAE 都很大**：CR≈4 时 990–5054 DN（远超 JPEG2000 的 16 DN）。这证实了学习型方法没有峰值误差约束——它们优化平均误差，少数像素可以偏差极大。
- **KLT+DWT 介于两者之间**：PAE=141 DN（nc=28 cr=1），比 JPEG2000 差但远好于学习型。
- **PAE 对科学应用至关重要**：单像素光谱反演（矿物识别、目标检测）对极端误差敏感。学习型方法在此类应用中存在根本性局限。

---

### 3.4 率失真曲线：CR vs MSE（有损/近无损方法）

![CR vs MSE](results/plots/cr_vs_mse.pdf)

**关键发现**：
- **MSE 趋势与 PSNR 一致**（两者是单调转换）——但 MSE 在对数坐标下更能体现数量级差异。
- **JPEG2000 在低 CR 压倒性优势**：CR=4 时 MSE=9 vs HyCASS MSE=1197 — 差距 133×。
- **SSCNet 在所有 CR 下 MSE 都极大**（CR=4 时 273,629）——确认该架构不适合本任务。
- **高 CR 区 HyCASS 主导**：CR>100 时 HyCASS 是唯一有实际意义的选项，MSE 控制在 77K–362K。

---

## 4. 综合对比表

### 4.1 CR ≈ 4（低压缩率，高质量区间）

| 方法 | CR | PSNR (dB) | PAE (DN) | MSE | Throughput (MB/s) |
|---|---|---|---|---|---|
| **JPEG2000 MCT** | 4.0 | **70.8** | **16** | **9** | 3.9 |
| KLT+DWT nc=28 | 4.3 | 64.4 | 141 | 37 | 7.4 |
| CAE1D | 4.0 | 52.1 | 1,024 | 643 | 38.8 |
| HyCASS | 4.0 | 49.5 | 990 | 1,197 | **179.2** |
| HYCOT | 4.0 | 45.8 | 1,364 | 2,855 | 35.3 |
| CAE3D | 4.0 | 31.7 | 2,898 | 70,060 | 108.7 |
| SSCNet | 4.0 | 25.8 | 5,054 | 273,629 | 153.9 |

### 4.2 CR ≈ 16（中等压缩率）

| 方法 | CR | PSNR (dB) | PAE (DN) | MSE | Throughput (MB/s) |
|---|---|---|---|---|---|
| **JPEG2000** | 16.0 | **48.9** | **377** | **1,358** | 4.6 |
| CAE1D | 13.9 | **47.6** | 1,212 | 1,788 | 10.7 |
| KLT+DWT nc=28 | 13.9 | 46.6 | 258 | 2,312 | 7.0 |
| HYCOT | 13.9 | 44.0 | 1,435 | 4,443 | 35.8 |
| HyCASS | 18.5 | 43.5 | 1,460 | 4,935 | **183.2** |
| CAE3D | 15.8 | 26.9 | 4,533 | 212,017 | 113.5 |
| SSCNet | 16.0 | 25.1 | 5,319 | 317,891 | 139.7 |

---

## 5. 结论

### 5.1 方法推荐矩阵

| 应用场景 | 推荐方法 | 原因 |
|---|---|---|
| **无损存储** | JPEG2000 MCT | CR=2.3×，损失=0，比 CCSDS-123 快 3× |
| **高质量有损（CR<8）** | JPEG2000 MCT | PSNR 碾压学习型（差 20+ dB），PAE 仅 16–85 DN |
| **中等压缩（CR=8–25）** | KLT+DWT 或 CAE1D | 经典方法和学习型差距缩小 |
| **高压缩（CR>100）** | HyCASS | 唯一可用选项，PSNR 仍有 24–31 dB |
| **实时/星载** | 暂无理想方案 | 学习型需 GPU，经典方法 CPU 吞吐不够 |

### 5.2 开放问题

1. **CCSDS-123 近无损（B-2）**：BrianShTsoi 实现不支持 M>0，需自行添加量化步长逻辑
2. **学习型方法 PAE 过大**：缺少误差边界约束，对科学应用不够安全
3. **训练稳定性**：CAE3D、HYCOT 存在非单调 RD 曲线（独立训练未收敛到一致最优点）
4. **跨数据集泛化**：预训练权重绑定 Berlin-Urban-Gradient，其他数据集需重训
5. **PSNR 基准**：当前用 peak=10000（数据实际峰值），论文结果通常用 peak=65535（16-bit 满量程）— 需统一

---

*报告生成时间: 2026-06-17 | Benchmark 代码: `/data/cyl/space_compression/hycass/benchmark/`*
