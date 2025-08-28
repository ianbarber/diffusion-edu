"""
Diffusion evolution — PyTorch mini‑lab
-------------------------------------------------
This single file shows *just enough* PyTorch code to connect the dots across:

A. Training targets
   - DDPM epsilon‑prediction (score matching via noise prediction)
   - v‑prediction (EDM/Imagen style parameterization)
   - Rectified Flow / Flow Matching (predict a velocity field)

B. Architectures
   - Tiny UNet (latent diffusion style)
   - Tiny DiT (Vision Transformer over latent patches)
   - Tiny MMDiT (two‑stream image+text with cross‑attention; both streams updated)

Design goals
------------
• Minimal, readable, hackable. No external deps beyond PyTorch.
• Continuous‑time noise schedule (cosine/EDM‑like) so eps/v/x0 are cleanly related.
• Fully interchangeable: the same training step works with UNet/DiT/MMDiT and
  eps/v/x0/flow targets.

You can wire this to any dataset. For “latent diffusion,” plug a VAE encoder/decoder
(see the VAE stubs) and train on latents. For a quick start, treat inputs as latents.

NOTE: This is a teaching scaffold, not a SOTA trainer.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------
# Utilities: time embeddings, patching, schedules
# ---------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding for continuous t in [0,1]. Shape: (B, dim).
    Uses the standard formulation from 'Attention is All You Need' and Stable Diffusion.
    """
    device = t.device
    half = dim // 2
    # Standard frequency formulation: 1 / (10000^(i/half))
    freqs = 1.0 / (10000.0 ** (torch.arange(half, device=device) / half))
    angles = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

@dataclass
class Schedule:
    """Cosine/EDM‑like schedule over t in [0,1]: alpha^2 + sigma^2 = 1.
    alpha(t) = cos(pi/2 * t), sigma(t) = sin(pi/2 * t).
    This keeps math tidy for eps/v/x0 conversions.
    """
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid numerical issues near t=1 where cos(π/2) → 0
        return torch.clamp(torch.cos(0.5 * math.pi * t), min=1e-8)
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * math.pi * t)

S = Schedule()

# Forward process helper for DDPM‑style training (continuous time)
@torch.no_grad()
def add_noise(x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return x_t = alpha x0 + sigma eps, with eps ~ N(0,I)."""
    eps = torch.randn_like(x0)
    a, s = S.alpha(t)[:, None, None, None], S.sigma(t)[:, None, None, None]
    x_t = a * x0 + s * eps
    return x_t, eps, a, s

# Conversions between parameterizations
# Given x_t = a x0 + s eps, and v = a eps - s x0, with a^2 + s^2 = 1

def v_from(x0, eps, a, s):
    return a * eps - s * x0

def x0_from(xt, v, a, s):
    # x0 = a * x_t - s * v
    return a * xt - s * v

def eps_from(xt, v, a, s):
    # eps = s * x_t + a * v
    return s * xt + a * v

# ---------------------------------------------
# VAE stubs for Latent Diffusion (optional)
# ---------------------------------------------
class TinyVAE(nn.Module):
    def __init__(self, in_ch=3, latent_ch=4):
        super().__init__()
        # Super tiny; replace with a real VAE for actual latent diffusion.
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, latent_ch, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_ch, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, in_ch, 3, 1, 1), nn.Tanh(),
        )
    def encode(self, x):
        return self.enc(x)
    def decode(self, z):
        return self.dec(z)

# ---------------------------------------------
# Architecture 1: Tiny UNet (latent‑space friendly)
# ---------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.emb = nn.Linear(emb_dim, ch)
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch*3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads
    def forward(self, x):
        B,C,H,W = x.shape
        h = self.norm(x)
        q,k,v = self.qkv(h).chunk(3, dim=1)
        # reshape to (B, heads, HW, C/heads)
        def shape(t):
            return t.view(B, self.num_heads, C//self.num_heads, H*W).transpose(2,3)
        q,k,v = shape(q), shape(k), shape(v)
        attn = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(q.shape[-1]))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(2,3).contiguous().view(B, C, H, W)
        return x + self.proj(out)

class UNetTiny(nn.Module):
    def __init__(self, in_ch=4, base=64, emb_dim=256, out_ch=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_emb_dim = emb_dim // 2  # Store the time embedding dimension
        self.time = nn.Sequential(nn.Linear(self.time_emb_dim, emb_dim*4), nn.SiLU(), nn.Linear(emb_dim*4, emb_dim))
        self.cond = nn.Linear(emb_dim, emb_dim)  # optional text/global cond (provide zeros if none)

        self.in_conv = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.down1 = ResBlock(base, emb_dim)
        self.pool1 = nn.Conv2d(base, base, 4, 2, 1)
        self.down2 = ResBlock(base, emb_dim)

        self.mid1 = ResBlock(base, emb_dim)
        self.mid_attn = AttnBlock(base)
        self.mid2 = ResBlock(base, emb_dim)

        self.up1 = ResBlock(base, emb_dim)
        self.up2 = ResBlock(base, emb_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, out_ch, 3, 1, 1)

    def forward(self, x, t, cond_emb=None):
        # x: (B,C,H,W), t: (B,), cond_emb: (B,emb_dim) or None
        temb = timestep_embedding(t, self.time_emb_dim)
        temb = self.time(temb)
        cemb = self.cond(cond_emb if cond_emb is not None else torch.zeros_like(temb))
        emb = temb + cemb

        h = self.in_conv(x)
        h1 = self.down1(h, emb)
        h2 = self.pool1(h1)
        h2 = self.down2(h2, emb)

        m = self.mid1(h2, emb)
        m = self.mid_attn(m)
        m = self.mid2(m, emb)

        u = self.up1(m, emb)
        u = F.interpolate(u, scale_factor=2, mode='nearest')
        u = self.up2(u + h1, emb)
        out = self.out(F.silu(self.out_norm(u)))
        return out

# ---------------------------------------------
# Architecture 2: Tiny DiT (Transformer over latent patches)
# ---------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=4, embed_dim=256, patch=2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.patch = patch
    def forward(self, x):
        x = self.proj(x)  # (B, D, H', W')
        B,D,H,W = x.shape
        tokens = x.flatten(2).transpose(1,2)  # (B, N, D)
        return tokens, (H,W)

class PatchUnembed(nn.Module):
    def __init__(self, out_ch=4, embed_dim=256, patch=2):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch, stride=patch)
    def forward(self, tokens, hw):
        B,N,D = tokens.shape
        H,W = hw
        x = tokens.transpose(1,2).view(B, D, H, W)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim)
        )
    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class CrossTransformerBlock(nn.Module):
    """Cross‑attention: query=image tokens, key/value = text tokens."""
    def __init__(self, dim_img, dim_txt, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.q_proj = nn.Linear(dim_img, dim_img)
        self.k_proj = nn.Linear(dim_txt, dim_img)
        self.v_proj = nn.Linear(dim_txt, dim_img)
        self.attn = nn.MultiheadAttention(dim_img, heads, batch_first=True)
        self.ln_q = nn.LayerNorm(dim_img)
        self.ln = nn.LayerNorm(dim_img)
        self.mlp = nn.Sequential(
            nn.Linear(dim_img, int(dim_img*mlp_ratio)), nn.GELU(), nn.Linear(int(dim_img*mlp_ratio), dim_img)
        )
    def forward(self, x_img, x_txt):
        q = self.q_proj(self.ln_q(x_img))
        k = self.k_proj(x_txt)
        v = self.v_proj(x_txt)
        x = x_img + self.attn(q, k, v, need_weights=False)[0]
        x = x + self.mlp(self.ln(x))
        return x

class DiTTiny(nn.Module):
    def __init__(self, in_ch=4, embed_dim=256, depth=6, patch=2, out_ch=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_emb_dim = embed_dim // 2  # Store the time embedding dimension
        self.pe = nn.Parameter(torch.zeros(1, 1024, embed_dim))  # simple, oversized
        nn.init.trunc_normal_(self.pe, std=0.02)
        self.time = nn.Sequential(nn.Linear(self.time_emb_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        self.cond = nn.Linear(embed_dim, embed_dim)
        self.patch = PatchEmbed(in_ch, embed_dim, patch)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(depth)])
        self.unpatch = PatchUnembed(out_ch, embed_dim, patch)
        self.head = nn.LayerNorm(embed_dim)
    def forward(self, x, t, cond_emb=None):
        tokens, hw = self.patch(x)
        B,N,D = tokens.shape
        pe = self.pe[:, :N, :]
        temb = self.time(timestep_embedding(t, self.time_emb_dim))
        cemb = self.cond(cond_emb if cond_emb is not None else torch.zeros_like(temb))
        tokens = tokens + pe + (temb + cemb)[:, None, :]
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.head(tokens)
        out = self.unpatch(tokens, hw)
        return out

# ---------------------------------------------
# Architecture 3: Tiny MMDiT (two‑stream image+text, both updated)
# ---------------------------------------------
class MMDiTTiny(nn.Module):
    def __init__(self, in_ch=4, img_dim=256, txt_dim=256, depth=6, patch=2, out_ch=4):
        super().__init__()
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.time_emb_dim = max(img_dim, txt_dim) // 2  # Use the larger of the two dims
        # Image side
        self.patch = PatchEmbed(in_ch, img_dim, patch)
        self.img_pe = nn.Parameter(torch.zeros(1, 1024, img_dim))
        nn.init.trunc_normal_(self.img_pe, std=0.02)
        # Text side (assume external text encoder produced tokens (B,L,txt_dim))
        self.txt_ln = nn.LayerNorm(txt_dim)
        self.txt_pe = nn.Parameter(torch.zeros(1, 256, txt_dim))
        nn.init.trunc_normal_(self.txt_pe, std=0.02)

        self.time_img = nn.Sequential(nn.Linear(self.time_emb_dim, img_dim), nn.SiLU(), nn.Linear(img_dim, img_dim))
        self.time_txt = nn.Sequential(nn.Linear(self.time_emb_dim, txt_dim), nn.SiLU(), nn.Linear(txt_dim, txt_dim))

        self.img_blocks = nn.ModuleList([TransformerBlock(img_dim) for _ in range(depth)])
        self.txt_blocks = nn.ModuleList([TransformerBlock(txt_dim) for _ in range(depth)])
        self.cross_i2t = nn.ModuleList([CrossTransformerBlock(img_dim, txt_dim) for _ in range(depth)])
        self.cross_t2i = nn.ModuleList([CrossTransformerBlock(txt_dim, img_dim) for _ in range(depth)])

        self.unpatch = PatchUnembed(out_ch, img_dim, patch)
        self.head = nn.LayerNorm(img_dim)

    def forward(self, x_img, t, txt_tokens):
        # x_img: (B,C,H,W), txt_tokens: (B,L,txt_dim)
        img_tokens, hw = self.patch(x_img)
        B,Ni,Di = img_tokens.shape
        Lt,Dt = txt_tokens.shape[1:]

        img_tokens = img_tokens + self.img_pe[:, :Ni, :]
        txt_tokens = self.txt_ln(txt_tokens + self.txt_pe[:, :Lt, :])

        temb_i = self.time_img(timestep_embedding(t, self.time_emb_dim))[:, None, :]
        temb_t = self.time_txt(timestep_embedding(t, self.time_emb_dim))[:, None, :]
        img_tokens = img_tokens + temb_i
        txt_tokens = txt_tokens + temb_t

        for bi in range(len(self.img_blocks)):
            # intra‑stream updates
            img_tokens = self.img_blocks[bi](img_tokens)
            txt_tokens = self.txt_blocks[bi](txt_tokens)
            # cross updates in both directions
            img_tokens = self.cross_i2t[bi](img_tokens, txt_tokens)
            txt_tokens = self.cross_t2i[bi](txt_tokens, img_tokens)

        img_tokens = self.head(img_tokens)
        out = self.unpatch(img_tokens, hw)
        return out

# ---------------------------------------------
# Training targets: eps/v/x0 and Flow Matching
# ---------------------------------------------
class TargetMode:
    EPS = 'eps'
    V = 'v'
    X0 = 'x0'
    FLOW = 'flow'  # rectified flow / flow matching

@dataclass
class TrainConfig:
    target: str = TargetMode.EPS  # 'eps' | 'v' | 'x0' | 'flow'
    lr: float = 1e-4
    cfg_dropout_prob: float = 0.1  # Classifier-free guidance dropout probability

class DiffusionLearner(nn.Module):
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def loss(self, x0, cond=None):
        B = x0.size(0)
        device = x0.device
        t = torch.rand(B, device=device)
        
        # Classifier-free guidance: randomly drop conditioning during training
        if cond is not None and self.training and self.cfg.cfg_dropout_prob > 0:
            # Create a mask for which samples to drop conditioning
            drop_mask = torch.rand(B, device=device) < self.cfg.cfg_dropout_prob
            if drop_mask.any():
                # Zero out conditioning for dropped samples (null conditioning)
                cond_shape = cond.shape
                if len(cond_shape) == 2:  # (B, D)
                    cond = cond * (~drop_mask).float().unsqueeze(1)
                elif len(cond_shape) == 3:  # (B, L, D) for sequence conditioning
                    cond = cond * (~drop_mask).float().unsqueeze(1).unsqueeze(2)

        if self.cfg.target == TargetMode.FLOW:
            # Rectified Flow / Flow Matching
            z = torch.randn_like(x0)
            xt = (1.0 - t)[:, None, None, None] * x0 + t[:, None, None, None] * z
            target_v = z - x0  # constant along the straight path
            pred_v = self.model(xt, t, cond)
            return F.mse_loss(pred_v, target_v)
        else:
            xt, eps, a, s = add_noise(x0, t)
            if self.cfg.target == TargetMode.EPS:
                target = eps
            elif self.cfg.target == TargetMode.V:
                target = v_from(x0, eps, a, s)
            elif self.cfg.target == TargetMode.X0:
                target = x0
            else:
                raise ValueError('Unknown target')
            pred = self.model(xt, t, cond)
            return F.mse_loss(pred, target)

# ---------------------------------------------
# Samplers: DDIM‑style for eps/v/x0; Euler ODE for Flow
# ---------------------------------------------
@torch.no_grad()
def ddim_sampler(model, x_shape, steps=50, cond=None, target=TargetMode.EPS, device='cuda', cfg_scale=1.0):
    """DDIM sampler with classifier-free guidance support.
    cfg_scale: Classifier-free guidance scale. 1.0 = no guidance, 7.5 = strong guidance (SD default)
    """
    B,C,H,W = x_shape
    x = torch.randn(B,C,H,W, device=device)
    # Start from 0.999 instead of 1.0 to avoid numerical instability
    ts = torch.linspace(0.999, 0.0, steps+1, device=device)
    for i in range(steps):
        t = ts[i].expand(B)
        tn = ts[i+1].expand(B)
        a, s = S.alpha(t)[:,None,None,None], S.sigma(t)[:,None,None,None]
        an, sn = S.alpha(tn)[:,None,None,None], S.sigma(tn)[:,None,None,None]
        
        # Classifier-free guidance
        if cfg_scale != 1.0 and cond is not None:
            # Get conditional and unconditional predictions
            pred_cond = model(x, t, cond)
            pred_uncond = model(x, t, None)  # Null conditioning
            # Apply guidance: pred = uncond + scale * (cond - uncond)
            pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = model(x, t, cond)
        if target == TargetMode.EPS:
            eps = pred
            x0 = (x - s * eps) / torch.clamp(a, min=1e-7)  # Extra safety clamp
        elif target == TargetMode.V:
            v = pred
            x0 = x0_from(x, v, a, s)
            eps = eps_from(x, v, a, s)
        elif target == TargetMode.X0:
            x0 = pred
            eps = (x - a * x0) / (s + 1e-8)
        else:
            raise ValueError('Use rf_sampler for flow matching.')
        # Deterministic DDIM update
        x = an * x0 + sn * eps
    return x

@torch.no_grad()
def rf_sampler(model, x_shape, steps=50, cond=None, device='cuda', cfg_scale=1.0):
    """Euler integrator for the ODE dx/dt = v_theta(x,t), from t=1 -> 0.
    
    Args:
        model: The flow matching model
        x_shape: Shape of samples to generate
        steps: Number of integration steps
        cond: Optional conditioning (for conditional generation)
        device: Device to run on
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
    """
    B,C,H,W = x_shape
    x = torch.randn(B,C,H,W, device=device)
    ts = torch.linspace(1.0, 0.0, steps+1, device=device)
    
    # Prepare null conditioning for CFG if needed
    if cond is not None and cfg_scale > 1.0:
        # Double the batch for conditional + unconditional
        x = torch.cat([x, x], dim=0)
        null_cond = torch.zeros_like(cond)
        cond_combined = torch.cat([cond, null_cond], dim=0)
    else:
        cond_combined = cond
    
    for i in range(steps):
        if cond is not None and cfg_scale > 1.0:
            # Process conditional and unconditional together
            t = ts[i].expand(B * 2)
            v_combined = model(x, t, cond_combined)
            v_cond, v_uncond = v_combined.chunk(2, dim=0)
            # Apply CFG
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            # Update only the conditional batch
            dt = (ts[i+1] - ts[i]).abs()
            x_cond, x_uncond = x.chunk(2, dim=0)
            x_cond = x_cond - dt * v  # integrate backward
            x = torch.cat([x_cond, x_uncond], dim=0)
        else:
            # Standard flow without CFG
            t = ts[i].expand(B)
            dt = (ts[i+1] - ts[i]).abs()
            v = model(x if cond is None else x[:B], t, cond_combined)
            if cond is not None and cfg_scale > 1.0:
                x = torch.cat([x[:B] - dt * v, x[B:]], dim=0)
            else:
                x = x - dt * v  # integrate backward (dt negative), hence minus
    
    # Return only the conditional batch if using CFG
    if cond is not None and cfg_scale > 1.0:
        return x[:B]
    return x

# ---------------------------------------------
# Minimal text conditioner (for DiT/MMDiT demos)
# ---------------------------------------------
class TinyTextToker(nn.Module):
    """Embeds a list of token ids or random ids as (B,L,dim).
    Replace with a real text encoder (e.g., CLIP) for serious use.
    """
    def __init__(self, vocab=1000, dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
    def forward(self, ids: torch.Tensor):
        return self.emb(ids)

# ---------------------------------------------
# Example wiring
# ---------------------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Example “latent” shapes (e.g., a 32x32 latent with 4 channels)
    B, C, H, W = 8, 4, 32, 32
    dummy_data = torch.randn(B, C, H, W, device=device)

    # --- Choose an architecture ---
    unet = UNetTiny(in_ch=C, out_ch=C).to(device)
    dit = DiTTiny(in_ch=C, out_ch=C).to(device)
    mmd = MMDiTTiny(in_ch=C, out_ch=C).to(device)
    arch = unet

    # --- Choose a target & learner ---
    cfg = TrainConfig(target=TargetMode.V)
    learner = DiffusionLearner(arch, cfg).to(device)

    # --- Optional: text conditioning (DiT/MMDiT) ---
    # txt = TinyTextToker().to(device)
    # txt_ids = torch.randint(0, 1000, (B, 32), device=device)
    # txt_tokens = txt(txt_ids)

    # --- One forward/backward step (illustrative) ---
    opt = torch.optim.AdamW(learner.parameters(), lr=cfg.lr)
    opt.zero_grad(set_to_none=True)
    loss = learner.loss(dummy_data)  # for DiT/MMDiT pass cond=txt_tokens.mean(1) or txt_tokens
    loss.backward()
    opt.step()
    print('Loss:', float(loss))

    # --- Sampling demo ---
    with torch.no_grad():
        # For eps/v/x0 targets:
        samples = ddim_sampler(learner.model, (4,C,H,W), steps=10, target=cfg.target, device=device)
        print('DDIM sample shape:', samples.shape)
        # For Flow Matching targets, initialize a learner with FLOW and use rf_sampler
        #fm = DiffusionLearner(UNetTiny(in_ch=C, out_ch=C).to(device), TrainConfig(target=TargetMode.FLOW)).to(device)
        #fm_samples = rf_sampler(fm.model, (4,C,H,W), steps=10, device=device)
        #yprint('RF sample shape:', fm_samples.shape)
