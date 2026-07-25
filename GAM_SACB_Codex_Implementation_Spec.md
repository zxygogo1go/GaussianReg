# Codex 实现任务：GAM-SACB-Net（头颈配准双模块版）

> 本文档的双模块修订版是当前实现依据。原 AGAM、GDCF、GAC-SACB 不再作为三个并列创新点，而是分别归并到 GACM 与 GCDR。双向一致性、自适应搜索、各向异性流正则和微分同胚积分留作后续研究机制，不属于本轮实现范围。

## 1. 目标

在 `x-xc/SACB_Net` 上新增最终模型 `GAM_SACB_Net`，保留原 SACB-Net 的 Encoder、SACB 金字塔、`cross_Sim`、SpatialTransformer 和逐级 flow composition；新增两个论文级核心模块：

1. **GACM — Gaussian Anatomy Correspondence Module**：位于 1/16 与 1/8 尺度，包含共享 Gaussian tokenizer、分离的位置/协方差代价、visibility-aware unbalanced transport、anchor correspondence 与体素化几何上下文。
2. **GCDR — Geometry-Conditioned Dense Registration Module**：包含 Gaussian–dense flow fusion、Gaussian geometry-conditioned SACB 和 full-resolution context refinement。

不得删除原 `SACB_Net`；新增模型必须能加载原 SACB-Net checkpoint，`strict=False` 时共享参数无尺寸冲突。

---

## 2. 文件修改

```text
新增 gaussian_anatomy.py
新增 geometry_conditioned_registration.py
新增 model_gam.py
新增 dataset/head_neck.py
新增 metrics.py
新增 train_gam.py
新增 evaluate_gam.py
新增 prepare_hntsmrg24.py
新增 tests/test_gaussian_anatomy.py
新增 tests/test_model_gam.py
修改 SACB1.py
修改 utils.py
修改 losses.py
修改 train.py
```

不引入 PyTorch 之外的新依赖。

---

## 3. 模块一：GACM（`gaussian_anatomy.py`）

### 3.1 数据结构

```python
@dataclass
class GaussianTokenSet:
    mu: torch.Tensor          # [B, N, 3]，归一化坐标，顺序 (D,H,W)，范围 [-1,1]
    cov: torch.Tensor         # [B, N, 3, 3]，SPD
    feat: torch.Tensor        # [B, N, E]
    anatomy: torch.Tensor     # [B, N, K]，softmax
    visibility: torch.Tensor  # [B, N, 1]，范围 [0.2,1]
    attention: Optional[torch.Tensor]  # [B,N,V]，仅训练/return_aux 时保存
```

### 3.2 `AnisotropicGaussianTokenizer3D`

构造函数：

```python
AnisotropicGaussianTokenizer3D(
    in_ch: int,
    token_dim: int = 64,
    num_tokens: int = 128,
    num_types: int = 8,
    temperature: float = 0.10,
    sigma_min: float = 0.015,
    sigma_max: float = 0.35,
)
```

实现：

1. `key_proj = Conv3d(in_ch, token_dim, 1)`。
2. `value_proj = Conv3d(in_ch, token_dim, 1)`。
3. 学习参数：
   - `queries: [N,E]`；
   - `anchors: [N,3]`，表示经过 `tanh` 前的 unconstrained 参数；以内部网格 `g∈[-0.95,0.95]^3` 的 `atanh(g)` 初始化，确保 `tanh(anchors)` 覆盖有效视野；
   - `log_radius: [N,3]`，初始化半径 0.35。
4. 展平体素特征为 `[B,V,E]`，构造归一化坐标 `coords:[V,3]`，顺序 `(D,H,W)`。
5. 计算：

```python
content_logits = einsum('ne,bve->bnv', normalize(queries), normalize(keys)) / sqrt(E)
radius = softplus(log_radius) + 0.05
spatial_bias = -0.5 * sum((coords - tanh(anchors))**2 / radius**2, dim=-1)
attention = softmax((content_logits + spatial_bias) / temperature, dim=-1)
```

6. 计算 token：

```python
mu = attention @ coords
feat = attention @ values
cov = sum_x attention * (x-mu)(x-mu)^T + 1e-5*I
```

7. 对 `cov` 执行 `torch.linalg.eigh`，特征值裁剪到 `[sigma_min**2, sigma_max**2]` 后重构 SPD。
8. `anatomy = softmax(type_head(feat), -1)`。
9. `visibility = 0.2 + 0.8 * sigmoid(vis_head(concat(feat, normalized_attention_entropy)))`。
10. 协方差、特征分解和 Sinkhorn 内部强制 float32；输出转换回输入 dtype。

### 3.3 Gaussian 距离

实现：

```python
pairwise_bures_wasserstein(
    mu_fixed, cov_fixed,
    mu_moving, cov_moving,
    chunk_size=32,
) -> tuple[
    Tensor[B,Nf,Nm],  # center_cost
    Tensor[B,Nf,Nm],  # covariance-only Bures cost
    Tensor[B,Nf,Nm],  # full Gaussian W2 cost
]
```

距离：

```text
||mu_f-mu_m||² + Tr(Sf + Sm - 2*(Sf^1/2 Sm Sf^1/2)^1/2)
```

矩阵平方根默认使用 float32 scaled Newton–Schulz 迭代；原因是接近各向同性的 token 会产生重复特征值，直接反向传播 `eigh` 的 eigenvector 梯度可能为 NaN。协方差谱界限可用 detached `eigvalsh` 计算 shift/scale 后施加，保持主 covariance 路径可微。按 fixed-token 维分块，禁止一次构造超大 `[B,Nf,Nm,3,3]`。匹配总代价中的 `C_cov` 只能使用 covariance-only 返回值，禁止重复计入中心位置距离。

### 3.4 `UnbalancedSinkhorn`

```python
UnbalancedSinkhorn(
    epsilon: float = 0.05,
    rho: float = 0.5,
    iterations: int = 25,
)
```

输入：

```python
cost: [B,Nf,Nm]
a: [B,Nf]  # fixed token mass
b: [B,Nm]  # moving token mass
```

使用 log-domain UOT：

```python
tau = rho / (rho + epsilon)
log_k = -cost / epsilon
u = tau * (log_a - logsumexp(log_k + v[:,None,:], dim=2))
v = tau * (log_b - logsumexp(log_k + u[:,:,None], dim=1))
P = exp(log_k + u[:,:,None] + v[:,None,:])
```

返回 `P:[B,Nf,Nm]`；行表示 fixed token，列表示 moving token。

### 3.5 `GaussianAnatomyMatcher3D`

构造函数：

```python
GaussianAnatomyMatcher3D(
    in_ch: int,
    spatial_size: tuple[int,int,int],
    num_tokens: int,
    token_dim: int = 64,
    num_types: int = 8,
    cost_feat: float = 1.0,
    cost_pos: float = 0.15,
    cost_cov: float = 0.25,
    cost_anatomy: float = 0.10,
    cost_visibility: float = 0.05,
)
```

moving 与 fixed 必须经过同一个 tokenizer 实例（权重完全共享），而不是两个独立 tokenizer；否则 feature/type cost 没有共同语义空间。

接口：

```python
result = matcher(moving_feat, fixed_feat, return_aux=True)
```

计算代价：

```python
C_feat = 1 - cosine(fixed.feat, moving.feat)
C_pos = pairwise squared distance of mu
C_cov = pairwise Bures covariance term
C_anatomy = 1 - fixed.anatomy @ moving.anatomy.T
C_visibility = -log(fixed.visibility * moving.visibility.T + 1e-6)
C = weighted sum
```

质量：

```python
a = normalize(fixed.visibility.squeeze(-1))
b = normalize(moving.visibility.squeeze(-1))
```

以 fixed token 为查询，计算：

```python
row_mass = P.sum(-1)
p_row = P / (row_mass[...,None] + 1e-6)
matched_mu_m = p_row @ moving.mu
anchor_disp_norm = matched_mu_m - fixed.mu
anchor_disp_voxel[...,0] = anchor_disp_norm[...,0] * (D-1)/2
anchor_disp_voxel[...,1] = anchor_disp_norm[...,1] * (H-1)/2
anchor_disp_voxel[...,2] = anchor_disp_norm[...,2] * (W-1)/2
entropy = -sum(p_row*log(p_row+1e-6)) / log(Nm)
anchor_conf = clamp(row_mass/(a+1e-6),0,1) * (1-entropy) * fixed.visibility
```

### 3.6 Gaussian rasterization

在 fixed 网格上，以 fixed token 的 `mu/cov` 为核进行分块 rasterization；输出：

```python
flow_g: [B,3,D,H,W]       # voxel displacement，SpatialTransformer 约定：fixed grid -> moving sampling location
confidence: [B,1,D,H,W]
context: [B,11,D,H,W]
```

`context` 通道固定为：

```text
0:3   Gaussian flow 的归一化位移，2*flow/(size-1)
3     confidence
4:10  fixed covariance / trace(cov) 的 6 个独立元素：zz,yy,xx,zy,zx,yx
10    anisotropy = (lambda_max-lambda_min)/(lambda_max+1e-6)
```

返回字典：

```python
{
    'flow': flow_g,
    'confidence': confidence,
    'context': context,
    'moving_tokens': moving_tokens,
    'fixed_tokens': fixed_tokens,
    'transport': P,
    'cost': C,
    'anchor_disp': anchor_disp_voxel,
    'anchor_conf': anchor_conf,
}
```

---

## 4. 模块二：GCDR（`geometry_conditioned_registration.py`）

实现：

```python
class GeometryConditionedDenseRegistrationBlock(nn.Module):
    def __init__(self, feat_ch, context_ch=11, hidden_ch=64, max_residual=1.0): ...
```

输入：

```python
moving_feat: [B,C,D,H,W]
fixed_feat: [B,C,D,H,W]
dense_flow: [B,3,D,H,W]      # 原 cross_Sim 输出
gaussian_flow: [B,3,D,H,W]
context: [B,11,D,H,W]
```

结构：

```python
dense_embed = ConvBlock(concat(fixed_feat, moving_feat, dense_flow), hidden_ch)
gaussian_embed = ConvBlock(concat(gaussian_flow, context), hidden_ch)
gate = sigmoid(GateHead(concat(dense_embed, gaussian_embed,
                                abs(dense_flow-gaussian_flow),
                                context[:,3:4])))
base_flow = gate*gaussian_flow + (1-gate)*dense_flow
residual = softsign(ResidualHead(concat(fixed_feat, moving_feat,
                                        dense_embed, gaussian_embed,
                                        dense_flow, gaussian_flow, context))) * max_residual
fused_flow = base_flow + residual
```

初始化：

- `GateHead` 最后一层 weight=0，bias=-4，使初始状态优先原 dense flow。
- `ResidualHead` 最后一层 weight=0，bias=0。

返回：

```python
fused_flow, gate
```

---

## 5. GCDR 的条件化 SACB：修改 `SACB1.py`

### 5.1 `SACB.__init__`

新增参数：

```python
cond_ch: int = 0
```

保持 `get_kernel/get_bias` 输入维度不变，保证原 checkpoint 可加载。

新增：

```python
self.cond_ch = cond_ch
self.cond_proj = nn.Linear(cond_ch, self._in_c) if cond_ch > 0 else None
```

`cond_proj.weight` 和 `cond_proj.bias` 全部初始化为 0。

### 5.2 `SACB.forward`

改为：

```python
def forward(self, x, cond=None):
```

要求：

1. `cond:[B,cond_ch,D,H,W]`，尺寸不同时 trilinear resize。
2. 对每个 batch 独立执行当前 KMeans；删除 `x_mean.squeeze(0)` 的 batch=1 假设。
3. 每个 cluster 计算：

```python
cond_centroid = cond_flat[b, mask].mean(0)
conditioned_centroid = feature_centroid + self.cond_proj(cond_centroid)
```

4. `conditioned_centroid` 送入原 `get_kernel/get_bias`。
5. 将 `torch.zeros(...).cuda()` 改为 `x.new_zeros(...)`。
6. `cond is None` 时行为与原实现完全一致。
7. 不修改原 SACB 的 KMeans、动态 kernel 和 residual 计算逻辑。

---

## 6. 修改 `utils.py`

`SpatialTransformer`：

1. 删除构造函数中的硬编码 `.to('cuda')`。
2. grid 在 CPU 创建后 `register_buffer('grid', grid.float())`，随模型自动移动。
3. 所有 `torch.meshgrid` 显式使用 `indexing='ij'`。
4. 保持 flow 通道顺序 `(D,H,W)` 和当前 grid_sample 反转逻辑不变。

---

## 7. 新增 `model_gam.py`

### 7.1 类

```python
class GAM_SACB_Net(nn.Module):
```

保留与原模型相同的以下属性名，便于载入 checkpoint：

```text
encoder
sacb_proj2
sacb_proj3
sacb_proj4
sacb_proj5
conv1
conv1_out
cross_sim
transformer
up_tri
```

新增：

```python
self.gam5 = GaussianAnatomyMatcher3D(16*c, size_1_16, num_tokens=128)
self.gam4 = GaussianAnatomyMatcher3D(16*c, size_1_8,  num_tokens=192)
self.fusion5 = GaussianDenseFusionBlock(16*c, 11, 64, max_residual=1.0)
self.fusion4 = GaussianDenseFusionBlock(16*c, 11, 64, max_residual=1.0)
```

四个 SACB 改为：

```python
SACB(..., cond_ch=11)
```

保留原 `conv1_out` 不变；新增：

```python
self.context_refiner = nn.Sequential(
    Conv3d(4*c + 3 + 11, 2*c, 3, padding=1),
    InstanceNorm3d(2*c), LeakyReLU(0.1),
    Conv3d(2*c, 2*c, 3, padding=1),
    InstanceNorm3d(2*c), LeakyReLU(0.1),
    Conv3d(2*c, 3, 3, padding=1),
)
```

`context_refiner` 最后一层全 0 初始化。

### 7.2 context resize

```python
def resize_context(ctx, target_size):
    return F.interpolate(ctx, size=target_size, mode='trilinear', align_corners=True)
```

context 的 flow 通道已经归一化，resize 时不乘尺度。

### 7.3 forward

```python
def forward(self, x, y, softsign_last=False, return_aux=False):
```

严格按以下顺序实现：

```python
M1,M2,M3,M4,M5 = self.encoder(x)
F1,F2,F3,F4,F5 = self.encoder(y)

# level 5, 1/16
G5 = self.gam5(M5, F5, return_aux=return_aux or self.training)
M5p = self.sacb_proj5(M5, G5['context'])
F5p = self.sacb_proj5(F5, G5['context'])
dense5 = self.cross_sim(M5p, F5p)
phi5_native, gate5 = self.fusion5(M5p, F5p, dense5, G5['flow'], G5['context'])
phi5 = self.up_tri(2.0 * phi5_native)

# level 4, 1/8
M4w = self.transformer[3](M4, phi5)
G4 = self.gam4(M4w, F4, return_aux=return_aux or self.training)
M4p = self.sacb_proj4(M4w, G4['context'])
F4p = self.sacb_proj4(F4,  G4['context'])
dense4 = self.cross_sim(M4p, F4p)
delta4, gate4 = self.fusion4(M4p, F4p, dense4, G4['flow'], G4['context'])
phi4_native = self.transformer[3](phi5, delta4) + delta4
phi4 = self.up_tri(2.0 * phi4_native)

# level 3, 1/4
ctx3 = resize_context(G4['context'], M3.shape[2:])
M3w = self.transformer[2](M3, phi4)
M3p = self.sacb_proj3(M3w, ctx3)
F3p = self.sacb_proj3(F3,  ctx3)
delta3 = self.cross_sim(M3p, F3p)
phi3_native = self.transformer[2](phi4, delta3) + delta3
phi3 = self.up_tri(2.0 * phi3_native)

# level 2, 1/2
ctx2 = resize_context(G4['context'], M2.shape[2:])
M2w = self.transformer[1](M2, phi3)
M2p = self.sacb_proj2(M2w, ctx2)
F2p = self.sacb_proj2(F2,  ctx2)
delta2 = self.cross_sim(M2p, F2p)
phi2_native = self.transformer[1](phi3, delta2) + delta2
phi2 = self.up_tri(2.0 * phi2_native)

# level 1, full resolution
ctx1 = resize_context(G4['context'], M1.shape[2:])
M1w = self.transformer[0](M1, phi2)
M1p = self.conv1(M1w)
F1p = self.conv1(F1)
delta1_base = self.conv1_out(torch.cat([F1p, M1p], dim=1))
delta1_ctx = self.context_refiner(torch.cat([F1p, M1p, delta1_base, ctx1], dim=1))
delta1 = delta1_base + F.softsign(delta1_ctx)
if softsign_last:
    delta1 = F.softsign(delta1)
Phi = self.transformer[0](phi2, delta1) + delta1
x_warped = self.transformer[0](x, Phi)
```

输出：

```python
if not return_aux:
    return x_warped, Phi
return {
    'warped': x_warped,
    'flow': Phi,
    'context_full': ctx1,
    'gam5': G5,
    'gam4': G4,
    'gate5': gate5,
    'gate4': gate4,
    'phi5_native': phi5_native,
    'phi4_native': phi4_native,
    'phi3_native': phi3_native,
    'phi2_native': phi2_native,
    'delta4': delta4,
}
```

### 7.4 checkpoint loader

实现：

```python
def load_sacb_checkpoint(self, path):
    state = torch.load(path, map_location='cpu')
    state = state.get('state_dict', state)
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    return self.load_state_dict(state, strict=False)
```

不得修改原共享层的参数尺寸。

---

## 8. 修改 `losses.py`

新增四个损失。

### 8.1 `GaussianTokenRegularization`

输入 `GaussianTokenSet`，返回：

```python
L_coverage = mean((attention.sum(dim=1) - N/V)**2)
L_repulsion = mean_offdiag(exp(-||mu_i-mu_j||² / 0.08))
L_compact = mean(trace(cov))
L_type = mean(entropy(anatomy)) - entropy(mean(anatomy over tokens))
L_visibility = (mean(visibility)-0.9)**2
L_token = L_coverage + 0.1*L_repulsion + 0.01*L_compact + 0.05*L_type + 0.1*L_visibility
```

### 8.2 `TransportCostLoss`

```python
L_transport = sum(P*C) / (sum(P)+1e-6)
```

分别计算 level 5 和 level 4 后取平均。

### 8.3 `AnchorFlowConsistencyLoss`

接口：

```python
loss(flow, fixed_mu, anchor_disp, anchor_conf)
```

使用 `grid_sample` 在 fixed token 的 `mu` 位置采样 flow；注意 `grid_sample` grid 顺序需由 `(D,H,W)` 转成 `(W,H,D)`；计算 confidence-weighted Charbonnier：

```python
sqrt((sampled_flow-anchor_disp)**2 + 1e-6)
```

应用：

```text
phi5_native 对齐 gam5.anchor_disp
delta4 对齐 gam4.anchor_disp
```

### 8.4 `AnisotropicFlowRegularization`（本轮暂缓）

该损失不进入本轮默认训练目标。Gaussian token 的 attention covariance 表示 token 空间支持域，并不等价于组织力学张量；在获得明确实验依据前，仅保留研究设想，不实现为论文主损失。

---

## 9. 修改 `train.py`

### 9.1 模型

```python
from model_gam import GAM_SACB_Net
model = GAM_SACB_Net(inshape=img_size, num_k=k).cuda()
```

如提供 baseline checkpoint，训练开始前执行 `model.load_sacb_checkpoint(...)`；不冻结任何层。

### 9.2 forward

```python
out = model(x, y, return_aux=True)
```

### 9.3 总损失

```python
L_sim = NCC_vxm()(out['warped'], y)
L_smooth = Grad3d(penalty='l2')(out['flow'], None)

L_deep = 0
for flow, scale_weight in [
    (out['phi4_native'], 0.05),
    (out['phi3_native'], 0.10),
    (out['phi2_native'], 0.15),
]:
    downsample x,y to flow.shape[2:]
    warp downsampled x with a SpatialTransformer of that size
    L_deep += scale_weight * NCC_vxm()(warped_x, downsampled_y)

L_token = 0.25 * (
    TokenReg(out['gam5']['moving_tokens']) +
    TokenReg(out['gam5']['fixed_tokens']) +
    TokenReg(out['gam4']['moving_tokens']) +
    TokenReg(out['gam4']['fixed_tokens'])
)

L_transport = 0.5 * (
    TransportLoss(out['gam5']['transport'], out['gam5']['cost']) +
    TransportLoss(out['gam4']['transport'], out['gam4']['cost'])
)

L_anchor = 0.5 * (
    AnchorLoss(out['phi5_native'],
               out['gam5']['fixed_tokens'].mu,
               out['gam5']['anchor_disp'],
               out['gam5']['anchor_conf']) +
    AnchorLoss(out['delta4'],
               out['gam4']['fixed_tokens'].mu,
               out['gam4']['anchor_disp'],
               out['gam4']['anchor_conf'])
)

loss = (
    1.00 * L_sim +
    0.30 * L_smooth +
    1.00 * L_deep +
    0.01 * L_token +
    0.02 * L_transport +
    0.05 * L_anchor
)
```

### 9.4 优化

```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```

允许 AMP；`eigh`、Bures、Sinkhorn 和 covariance rasterization 内部关闭 autocast并使用 float32。

TensorBoard 记录：

```text
L_sim, L_smooth, L_deep, L_token, L_transport, L_anchor
gate5.mean, gate4.mean
mean token visibility
mean transport mass
```

同时记录最终 flow 的负 Jacobian 比例、最小 Jacobian determinant 和 validation Dice，但 Jacobian 仅作为本轮监测指标，不包装成新增创新模块。

---

## 10. 测试

### `tests/test_gaussian_anatomy.py`

验证：

1. tokenizer 输出尺寸正确。
2. `cov` 全部有限且最小特征值 > 0。
3. Sinkhorn 输出非负、无 NaN/Inf。
4. matcher 输出 flow `[B,3,D,H,W]`、context `[B,11,D,H,W]`。
5. loss.backward 后 `queries`、`anchors`、`key_proj`、`type_head` 有非零梯度。

### `tests/test_model_gam.py`

使用 `inshape=(32,32,32), ch_scale=2, num_k=3`：

1. `return_aux=False` 返回 `(warped, flow)`。
2. `return_aux=True` 返回完整字典。
3. warped `[B,1,32,32,32]`，flow `[B,3,32,32,32]`。
4. 前向和反向无 NaN/Inf。
5. 原 SACB checkpoint 使用 `strict=False` 加载时，共享层无 shape mismatch。
6. 验证梯度到达 `gam5/gam4`、`fusion5/fusion4`、`cond_proj`、`context_refiner`。

---

## 11. 固定超参数

```python
TOKEN_DIM = 64
TOKEN_NUM_L5 = 128
TOKEN_NUM_L4 = 192
NUM_TYPES = 8
CONTEXT_CH = 11
SINKHORN_EPS = 0.05
SINKHORN_RHO = 0.5
SINKHORN_ITERS = 25
BURES_CHUNK = 32
RASTER_VOXEL_CHUNK = 32768
COST_FEAT = 1.0
COST_POS = 0.15
COST_COV = 0.25
COST_ANATOMY = 0.10
COST_VISIBILITY = 0.05
```

不得增加独立训练阶段、冻结阶段、外部分割器或测试时优化。数据读取仅允许增加医学图像 I/O 所必需的 `nibabel` 与 `SimpleITK`。

---

## 12. 头颈 HNTS-MRG24 数据与训练协议

### 12.1 主任务

- 主数据集为 HNTS-MRG24 的 150 个纵向患者。
- moving 为经过确定性 rigid/affine 预对齐的 preRT T2，fixed 为 midRT T2。
- 官方 deformably registered preRT 只允许用于 QA 对照，禁止作为模型输入，避免目标信息泄漏。
- mask 标签固定为 `0=background, 1=GTVp, 2=GTVn`。
- 每位患者只属于 train/validation/test 之一，禁止 patient leakage。

### 12.2 物理空间预处理

- 读取并验证 NIfTI affine、orientation、spacing 与 image/mask 一致性。
- 先进行 centered rigid，再进行 affine Mattes mutual-information 预对齐；affine 恶化或 determinant/translation 越界时回退 rigid。
- 以 fixed 图像物理中心构建统一目标网格：默认 1.5 mm isotropic，网络数组 `(D,H,W)=(128,160,160)`。
- image 使用线性插值，mask 使用 nearest-neighbor；保存处理后的 `.npy`、原始/目标 affine、spacing、方向、预对齐参数和 QA 状态。
- MRI 使用非零区域 0.5–99.5 percentile robust clipping 后缩放到 `[0,1]`。
- 任何肿瘤 mask 超出目标视野的病例必须进入 QA 失败列表，不得静默裁掉后用于指标。

### 12.3 划分和指标

- 默认固定随机种子 2026，按患者做 80/10/10 分层划分，并按 midRT 的 GTVp/GTVn presence 分层。
- 必报：mean/median per-class Dice、HD95、ASSD、负 Jacobian ratio、`detJ<0.5` ratio、minimum detJ。
- Dice/表面距离仅统计 moving 与 fixed 两个时间点都存在的类别；完全缓解的肿瘤类别不作为“无法配准”的错误惩罚。
- 如数据提供可靠 landmark，再额外报告 TRE；不得从分割质心伪造 landmark TRE。

### 12.4 A100 40 GB 默认设置

- 输入 `(128,160,160)`，batch size 1，AMP 开启，AdamW，gradient clipping 5.0。
- L5/L4 token 数默认 128/192；rasterization 与 Bures 必须分块并在 float32 中执行。
- 保存 latest/best checkpoint、完整配置、数据 manifest hash、逐病例验证结果和 TensorBoard 日志。
> **状态说明（minimal-v2）**
>
> 本文记录的是第一版双尺度 GACM/GCDR 设计，现仅作为历史设计与干预实验依据。
> 当前可训练模型已经根据干预结果精简为：单尺度 L4 Gaussian token
> correspondence + 轻量有界残差校正器。当前结构、配置和训练方式以
> `README.md`、`model_gam.py` 和 `configs/gam_sacb_hntsmrg24.json` 为准。
> 第一版 checkpoint 与 minimal-v2 不兼容。
