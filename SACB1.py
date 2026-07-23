import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple
from nn_util import get_act_layer, conv, unfoldNd
# import functools
from einops import rearrange, reduce
import numpy as np
try:
    from kmeans_gpu import KMeans
except ImportError:
    class KMeans:
        """Small deterministic PyTorch fallback used when kmeans_gpu is absent."""

        def __init__(
            self,
            n_clusters,
            max_iter=100,
            tolerance=1.0e-6,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=None,
        ):
            self.n_clusters = int(n_clusters)
            self.max_iter = min(int(max_iter), 25)
            self.tolerance = float(tolerance)

        def fit_predict(self, x):
            if x.ndim != 2 or x.shape[0] < self.n_clusters:
                raise ValueError('KMeans input must have at least n_clusters rows')
            indices = torch.linspace(0, x.shape[0] - 1, self.n_clusters, device=x.device).long()
            centroids = x[indices]
            closest = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            for _ in range(self.max_iter):
                distances = torch.cdist(x.float(), centroids.float())
                updated_closest = distances.argmin(dim=1)
                updated = []
                for cluster in range(self.n_clusters):
                    mask = updated_closest.eq(cluster)
                    updated.append(x[mask].mean(dim=0) if bool(mask.any()) else centroids[cluster])
                updated_centroids = torch.stack(updated, dim=0)
                delta = (updated_centroids.detach() - centroids.detach()).abs().max()
                closest = updated_closest
                centroids = updated_centroids
                if float(delta) <= self.tolerance:
                    break
            return closest, centroids

def tuple_(x, length = 1):
    return x if isinstance(x, tuple) else ((x,) * length)


class KM_GPU():
    def __init__(self,num_k=4, rng_seed=0, 
                 tol=1e-9, m_iter=1e9, fix_rng=True,
                 max_neighbors=160*192*224):
        super(KM_GPU, self).__init__()   
        self.seed = rng_seed
        self.fix_rng = fix_rng
        
        self.km = KMeans(
            n_clusters= num_k,
            max_iter= int(m_iter),
 
            tolerance=tol,
            distance='euclidean',
            sub_sampling=None,
            max_neighbors=max_neighbors,
        )
    def set_k(self, k):
        self.km.n_clusters = k
    def get_cluster_map(self, x):
        if self.fix_rng: np.random.seed(self.seed)
        closest, centroid = self.km.fit_predict(x)
        return closest, centroid
        
class cross_Sim(nn.Module): 
    def __init__(self, win_s=3):
        super(cross_Sim, self).__init__()
        self.wins = win_s
        self.win_len = win_s**3
              
    def forward(self, Fx, Fy, wins=None):
        if wins:
            self.wins = wins
            self.win_len = wins**3
        b, c, d, h, w = Fy.shape
      
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [self.wins] * 3]
        grid = torch.stack(torch.meshgrid(vectors, indexing='ij'), -1).type(torch.FloatTensor)
   
        G = grid.reshape(self.win_len, 3).unsqueeze(0).unsqueeze(0).to(device=Fx.device, dtype=Fx.dtype)

        Fy = rearrange(Fy, 'b c d h w -> b (d h w) 1 c')
        pd = self.wins // 2  # 1

        Fx = F.pad(Fx,  tuple_(pd, length=6)) 
     
        Fx = Fx.unfold(2, self.wins, 1).unfold(3, self.wins, 1).unfold(4, self.wins, 1)
        Fx = rearrange(Fx, 'b c d h w wd wh ww -> b (d h w) (wd wh ww) c')

        attn = (Fy @ Fx.transpose(-2, -1))
        sim = attn.softmax(dim=-1)
        out = (sim @ G) 
        out = rearrange(out , 'b (d h w) 1 c -> b c d h w', d=d,h=h,w=w)
    
        return out


class SACB(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride=1,
                 in_proj_n=1,
                 padding=1, dilation=1, groups=1,
                 num_k=4, 
                 act='prelu', residual=True, 
                 mean_type = 's',
                 scale_f=1,
                 n_mlp=1,
                 sample_n = 5,
                 m_iter=1e10,
                 tol   =1e-10,
                 fix_rng= False,
                 cond_ch=0,
                 ):
        super(SACB, self).__init__()
        self.ks       = ks
        self.stride   = stride
        self.padding  =  tuple(x for x in reversed(_triple(padding)) for _ in range(2))
        self.dilation = _triple(dilation)
        self.num_k    = num_k
        self.res      = residual
        self.out_ch = out_ch
        in_ch_n = int(in_ch * in_proj_n)
        self.w   = nn.Parameter(torch.Tensor(out_ch, in_ch_n // groups, self.ks**3))
        # self.b   = nn.Parameter(torch.Tensor(num_k, out_ch)) if bias else None
        self.act = get_act_layer(act) if act else None
        self.reset_parameters()
        self.scale_f = scale_f
        self.mean_type = mean_type
        self.km = KM_GPU(num_k=num_k, rng_seed=0, m_iter=m_iter, tol=tol, fix_rng=fix_rng)
        
        inner_dims = 128 * n_mlp
        inner_dims2 = 64 * n_mlp
        
        self.sample_n = sample_n
        if   mean_type =='s': 
            _in_c = in_ch_n
        elif mean_type =='c': _in_c = self.ks**3
        else: _in_c = in_ch + self.ks**3
        self._in_c = _in_c
        self.cond_ch = int(cond_ch)
        self.cond_proj = nn.Linear(self.cond_ch, self._in_c) if self.cond_ch > 0 else None
        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        
        self.get_kernel = nn.Sequential(
                nn.Linear(_in_c, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, inner_dims), nn.ReLU(),
                nn.Linear(inner_dims, self.ks**3), nn.Sigmoid()
                )
        
        self.get_bias = nn.Sequential(
                nn.Linear(in_features=_in_c,  out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=inner_dims2), nn.ReLU(),
                nn.Linear(in_features=inner_dims2, out_features=out_ch),
                )   
        # self.proj_in  = conv(in_ch, in_ch_n, 3, 1, 1, bias=False)
        self.proj_in  = conv(in_ch, in_ch_n, 3, 1, 1, act=act, norm='instance')
     
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        # if self.b is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(self.b, -bound, bound)
    
    def set_num_k(self, k):
        self.num_k = k
        self.km.set_k(k)    
               
    def scale(self, x, factor, mode='nearest'):
        if mode == 'nearest':
            return F.interpolate(x, scale_factor=factor, mode=mode)  
        else: 
            return F.interpolate(x, scale_factor=factor, mode='trilinear', align_corners=True)  
    
    def feat_mean(self, x, mean_type='s'):
        if   mean_type == 's': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
        elif mean_type == 'c': x = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
        else: 
            xs = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) c', 'mean')
            xc = reduce(x, 'b c nd nh nw k1 k2 k3 -> b (nd nh nw) (k1 k2 k3)', 'mean')
            x = torch.cat([xs, xc], -1)
        return x
       
    def forward(self, x, cond=None):
        b,c,d,h,w = x.shape
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError('cond was provided but this SACB was constructed with cond_ch=0')
            if cond.ndim != 5 or cond.shape[0] != b or cond.shape[1] != self.cond_ch:
                raise AssertionError('cond must have shape [B,cond_ch,D,H,W]')
            if cond.shape[2:] != (d, h, w):
                cond = F.interpolate(cond, size=(d, h, w), mode='trilinear', align_corners=True)
            cond_flat = rearrange(cond, 'b c d h w -> b (d h w) c')
        else:
            cond_flat = None
        x_in = x
        x = self.proj_in(x) 
        x_pad = F.pad(x, self.padding)
        x = x_pad.unfold(2,self.ks,self.stride).unfold(3,self.ks,self.stride).unfold(4,self.ks,self.stride)
        # x = unfoldNd(x, kernel_size=3, dilation=1, padding=1, stride=1)
       
        x_mean = self.feat_mean(x, self.mean_type)
        x = rearrange(x,'b c nd nh nw k1 k2 k3 -> b (c k1 k2 k3) (nd nh nw)') 
        out = x.new_zeros(b, self.out_ch, d*h*w)
        for batch_idx in range(b):
            # Clustering is numerically fragile in float16 and does not need
            # autocast savings. Keeping its input in float32 makes AMP usable
            # on A100 while centroids are cast back for the learned heads.
            cluster_idx, centroid = self.km.get_cluster_map(x_mean[batch_idx].float())
            if not torch.is_tensor(cluster_idx):
                cluster_idx = torch.as_tensor(cluster_idx, device=x.device)
            else:
                cluster_idx = cluster_idx.to(x.device)
            if not torch.is_tensor(centroid):
                centroid = torch.as_tensor(centroid, device=x.device, dtype=x.dtype)
            else:
                centroid = centroid.to(device=x.device, dtype=x.dtype)
            for i in range(self.num_k):
                mask = cluster_idx.eq(i)
                if not bool(mask.any()):
                    continue
                conditioned_centroid = centroid[i].unsqueeze(0)
                if cond_flat is not None:
                    cond_centroid = cond_flat[batch_idx, mask].mean(0, keepdim=True)
                    conditioned_centroid = conditioned_centroid + self.cond_proj(cond_centroid)
                weight = rearrange(self.get_kernel(conditioned_centroid), 'b k -> b 1 1 k') * self.w
                bias = rearrange(self.get_bias(conditioned_centroid), 'b o -> b o 1')
                # Each voxel belongs to exactly one KMeans cluster. The
                # original implementation evaluated every cluster-specific
                # convolution at every voxel and masked most responses only
                # afterwards, multiplying SACB compute by ``num_k``. Selecting
                # the cluster's patches first is algebraically identical and
                # keeps parameter/checkpoint shapes unchanged.
                selected_patches = x[batch_idx, :, mask]
                kernel = rearrange(weight, 'b o i k -> b o (i k)')[0]
                response = torch.einsum('o i, i s -> o s', kernel, selected_patches)
                response = response + bias[0]
                out[batch_idx, :, mask] = response
           
        out = rearrange(out, 'b o (d h w) -> b o d h w', d=d, h=h, w=w)
        if self.act: out = self.act(out)
        if self.res: out = out + x_in
     
        return out 
