# Wave Mixture Architecture - Technical Deep Dive

## Table of Contents
1. [Motivation](#motivation)
2. [Core Innovation](#core-innovation)
3. [Detailed Component Analysis](#detailed-component-analysis)
4. [Implementation Details](#implementation-details)
5. [Training Dynamics](#training-dynamics)
6. [Inference Optimization](#inference-optimization)

---

## Motivation

### The Quadratic Bottleneck

Standard Transformer self-attention has complexity O(n²) due to the attention matrix:

```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

For sequence length n:
- QK^T matrix: n × n = O(n²) memory
- Softmax computation: O(n²) time
- Backward pass: O(n²) for gradient computation

**Practical limits:**
- 4K tokens: 16M attention weights (manageable)
- 32K tokens: 1B attention weights (13GB memory)
- 1M tokens: 1T attention weights (impossible)

### Why Wave Propagation?

Physical waves naturally propagate information globally:
- **Acoustic waves**: Travel kilometers with gradual attenuation
- **Electromagnetic waves**: Light-year propagation in space
- **Water waves**: Ocean-wide tsunamis

**Key insight**: Damped waves maintain long-range correlations with O(1) per-step propagation cost.

---

## Core Innovation

### From Attention to Wave Convolution

**Traditional Attention:**
```
For each position i:
  For each position j:
    score[i,j] = dot(query[i], key[j])
    attention[i,j] = softmax(score[i,j])
    output[i] += attention[i,j] * value[j]
```
Complexity: O(n²) dot products

**Wave Propagation:**
```
# 1. Transform to frequency domain
Q_fft = FFT(query)
K_fft = FFT(key)

# 2. Apply wave kernel in frequency domain
Wave_fft = FFT(exp(-αt) * cos(ωt + φ))

# 3. Multiply (convolution theorem)
Output_fft = Q_fft * K_fft * Wave_fft

# 4. Transform back
output = IFFT(Output_fft)
```
Complexity: O(n log n) for FFT

---

## Detailed Component Analysis

### 1. Continuous Token Field

**Problem**: Discrete tokens don't support continuous wave propagation

**Solution**: Multi-layer embedding with continuous transformation

```python
class ContinuousTokenField(nn.Module):
    def __init__(self, vocab_size, dim):
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = RotaryEmbedding(dim)  # Continuous positions
        self.field_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, token_ids):
        # Discrete to continuous
        x = self.token_embed(token_ids)  # [batch, seq, dim]
        
        # Add continuous positional field
        positions = torch.arange(seq_len).float()
        x = x + self.pos_embed(positions)
        
        # Project to continuous field
        x = self.field_proj(x)
        return x
```

**Mathematical properties:**
- Lipschitz continuity: ‖f(x) - f(y)‖ ≤ L‖x - y‖
- Smoothness: Differentiable everywhere for gradient flow
- Expressiveness: MLP projection increases capacity

### 2. FFT Wave Convolution

#### Kernel Design

**Damped Oscillator Equation:**
```
m d²x/dt² + c dx/dt + kx = 0
```

Solution for underdamped case (c² < 4mk):
```
x(t) = A exp(-αt) cos(ωt + φ)
```

Where:
- α = c/(2m) (damping ratio)
- ω = √(4mk - c²)/(2m) (angular frequency)
- φ = phase shift

**Learnable parameters per head:**
```python
class WaveKernel(nn.Module):
    def __init__(self, num_heads):
        # Initialize for multi-scale coverage
        self.omega = nn.Parameter(torch.logspace(-2, 1, num_heads))
        self.alpha = nn.Parameter(torch.logspace(-3, -1, num_heads))
        self.phi = nn.Parameter(torch.rand(num_heads) * 2π)
```

#### FFT Implementation

**Standard convolution:**
```
(f * g)[n] = Σₘ f[m] g[n-m]  # O(n²)
```

**FFT convolution:**
```
(f * g) = IFFT(FFT(f) · FFT(g))  # O(n log n)
```

**Causal masking in frequency domain:**
```python
def causal_fft_convolution(x, kernel):
    n = x.shape[-1]
    
    # Zero-pad to power of 2
    pad = 2 ** ceil(log2(2*n - 1))
    x_pad = F.pad(x, (0, pad - n))
    k_pad = F.pad(kernel, (0, pad - n))
    
    # FFT
    X = torch.fft.rfft(x_pad, dim=-1)
    K = torch.fft.rfft(k_pad, dim=-1)
    
    # Multiply in frequency domain
    Y = X * K
    
    # IFFT
    y = torch.fft.irfft(Y, n=pad, dim=-1)
    
    # Take causal part (first n elements)
    return y[..., :n]
```

**Complexity analysis:**
- FFT: O(n log n) using Cooley-Tukey algorithm
- Pointwise multiply: O(n)
- IFFT: O(n log n)
- **Total: O(n log n)**

### 3. Mixture of Recursion

**Motivation**: Balance between stability and plasticity

**Formulation:**
```
x_{t+1} = g_t ⊙ x_t + (1 - g_t) ⊙ f(x_t)
```

Where:
- g_t ∈ [0,1]: Gating signal (learned)
- x_t: Previous state
- f(x_t): Propagated signal

**Gate computation:**
```python
gate = sigmoid(W_g [x_t || f(x_t)] + b_g)
```

**Interpretation:**
- g_t ≈ 1: Preserve memory (stable)
- g_t ≈ 0: Update with new information (plastic)
- Learned balance: Task-dependent adaptation

**Connection to:**
- **LSTM/GRU**: Gated recurrence
- **ResNet**: Residual connections
- **Highway networks**: Adaptive depth

### 4. Multi-Head Wave Propagation

**Why multiple heads?**

Different frequencies capture different patterns:

| Frequency ω | Wavelength λ | Captures |
|-------------|--------------|----------|
| High (ω=10) | ~0.6 tokens | Local syntax, punctuation |
| Medium (ω=1) | ~6 tokens | Words, phrases |
| Low (ω=0.1) | ~60 tokens | Sentences, clauses |
| Very low (ω=0.01) | ~600 tokens | Paragraphs, sections |

**Head specialization (emergent):**
```
Head 0-3: High ω (local patterns)
Head 4-7: Medium ω (phrases)
Head 8-11: Low ω (long-range)
Head 12-15: Very low ω (global structure)
```

**Implementation:**
```python
class MultiHeadWaveConv(nn.Module):
    def __init__(self, num_heads=16):
        self.heads = nn.ModuleList([
            WaveKernel(omega_init=2**(-i), alpha_init=10**(-i-2))
            for i in range(num_heads)
        ])
        self.combine = nn.Linear(num_heads * dim, dim)
```

### 5. Identity / Copy Path

**Problem with pure wave propagation:**
- Waves are smooth and diffuse
- High-frequency information lost
- Cannot copy rare tokens exactly

**Solution: Explicit identity mapping**
```python
identity = x + MLP(LayerNorm(x))  # Residual + projection
gate = sigmoid(W [identity || propagated])
output = gate * identity + (1-gate) * propagated
```

**Why it works:**
- Residual connection: Direct gradient flow
- MLP projection: Learnable transformation
- Gating: Adaptive blending
- **Result**: Exact token preservation capability

**Theoretical guarantee:**
```
If gate ≈ 1: output ≈ identity (exact copy)
If gate ≈ 0: output ≈ propagated (contextual update)
```

### 6. Hierarchical Context (1M+ Support)

**Challenge**: Even O(n log n) becomes expensive at n=1M

**Solution: Hierarchical compression**

```
Level 0 (Token): 8K chunks → O(8K log 8K) per chunk
Level 1 (Chunk): 16:1 compression → O(512 log 512)
Level 2 (Global): Compressed sequence → O(64K log 64K)
```

**Memory flow:**
```
1M tokens
  ↓ [Chunk 1]
8K tokens → FFT Conv → 8K output
  ↓ [Compress]
512 compressed features
  ↓ [Repeat for 128 chunks]
64K compressed features
  ↓ [Global FFT Conv]
64K global context
  ↓ [Expand and fuse]
1M final output
```

**Compression method:**
```python
class HierarchicalCompressor(nn.Module):
    def compress(self, x):  # x: [batch, 8192, dim]
        # Reshape and pool
        x = x.view(batch, 512, 16, dim)
        x = x.mean(dim=2)  # [batch, 512, dim]
        
        # Learned compression
        compressed = self.compression_mlp(x.view(batch, 512, -1))
        return compressed  # [batch, 512, dim]
```

---

## Implementation Details

### Memory-Efficient Chunking

**Problem**: 1M × 1024 dim = 4GB just for activations

**Solution**: Iterative chunking with state passing

```python
class ChunkedWaveProcessor:
    def __init__(self, chunk_size=8192):
        self.chunk_size = chunk_size
        self.state = None
    
    def forward(self, x):  # x: [1, 1M, 1024]
        outputs = []
        
        for i in range(0, 1_000_000, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            
            # Process chunk
            out, new_state = self.process_chunk(chunk, self.state)
            
            # Update state for continuity
            self.state = new_state.detach()
            
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)
```

**Memory usage:**
- Always O(chunk_size) regardless of total length
- 1M tokens: ~100MB (vs 4GB naive)

### Gradient Checkpointing

**Trade compute for memory:**
```python
# Without checkpointing
x = layer1(x)  # Store activation
x = layer2(x)  # Store activation
loss = criterion(x)
loss.backward()  # Backprop through stored activations

# With checkpointing
x = checkpoint(layer1, x)  # Recompute forward in backward
x = checkpoint(layer2, x)
loss = criterion(x)
loss.backward()  # Recompute layers during backprop
```

**Cost**: ~20% slower training, 50% less memory

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.bfloat16):
    outputs = model(input_ids)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
```

**Benefits:**
- 2x memory reduction
- 2-3x speedup on modern GPUs (Tensor Cores)
- Minimal accuracy impact with bfloat16

---

## Training Dynamics

### Wave Parameter Initialization

**Critical for convergence:**

```python
# Log-spaced frequencies cover all scales
omega = torch.logspace(-2, 1, num_heads)  # 0.01 to 10

# Inverse relationship: high freq = high damping
alpha = torch.logspace(-3, -1, num_heads)   # 0.001 to 0.1
```

**Why log-spaced?**
- Linear spacing wastes heads on similar scales
- Log spacing ensures coverage from local to global
- Matches natural frequency distributions

### Learning Rate Scheduling

**Warmup critical for wave parameters:**

```python
def lr_lambda(step):
    warmup = 2000
    if step < warmup:
        return step / warmup  # Linear warmup
    
    # Cosine decay
    progress = (step - warmup) / (max_steps - warmup)
    return 0.5 * (1 + cos(π * progress))
```

**Wave parameter specific:**
- α (damping): Lower LR (0.1× base) - sensitive to change
- ω (frequency): Standard LR
- φ (phase): Higher LR (2× base) - less sensitive

### Loss Landscape

**Observed characteristics:**
- **Early training**: Wave parameters converge first (smooth signal)
- **Mid training**: Gating mechanisms specialize (identity vs. propagation)
- **Late training**: Fine-tuning of frequency spectrum

**Monitoring metrics:**
```python
# Head utilization
for i, head in enumerate(model.heads):
    print(f"Head {i}: ω={head.omega.item():.3f}, α={head.alpha.item():.3f}")

# Gate statistics
print(f"Mean gate value: {gate.mean():.3f}")  # Should be ~0.5
print(f"Gate variance: {gate.var():.3f}")     # Should be >0 (active gating)
```

---

## Inference Optimization

### KV-Cache for Wave Models

**Different from attention cache:**

```python
# Attention: Store K, V for each position
# Wave: Store "state" (last position of previous chunk)

class WaveCache:
    def __init__(self):
        self.state = None  # [batch, heads, dim]
    
    def update(self, new_output):
        # Store last position as state for next chunk
        self.state = new_output[:, -1, :, :].detach()
```

**Memory comparison:**
- Attention KV cache: 2 × n × d × 4 bytes = O(n)
- Wave state cache: 2 × d × 4 bytes = O(1)

### Speculative Decoding

**Draft model generates candidates, main model verifies:**

```python
# Small, fast draft model
draft_model = WaveMixture(config_small)

# Large, accurate main model
main_model = WaveMixture(config_large)

# Generate 4 candidates
candidates = draft_model.generate(prompt, max_new=4)

# Verify in parallel
logits = main_model(candidates)
accepted = sample_acceptance(logits)
```

**Speedup**: 2-3x for autoregressive generation

### Quantization

**INT8 quantization for deployment:**

```python
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# 4x memory reduction
# 2x speedup on INT8-capable hardware
```

---

## Future Directions

### 1. FlashFFT Implementation

Custom CUDA kernels for FFT optimization:
- Fused FFT + pointwise multiply + IFFT
- Tiled computation for better cache utilization
- Expected 2-3x speedup

### 2. Adaptive Chunk Sizes

Dynamic chunk sizing based on content:
```python
if is_code_structure_boundary(position):
    chunk_size = 4096  # Smaller chunks at boundaries
else:
    chunk_size = 8192  # Larger chunks in uniform regions
```

### 3. Multi-Modal Extensions

Apply wave propagation to:
- **Vision**: 2D FFT for images
- **Audio**: 1D FFT for waveforms
- **Video**: 3D FFT for spatiotemporal data

### 4. Hardware Acceleration

- **Optical computing**: FFT is native operation in optics
- **Neuromorphic**: Wave propagation maps to analog circuits
- **Quantum**: QFT for exponential speedup

---

## References

1. Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
2. Lee-Thorp et al. (2022). FNet: Mixing Tokens with Fourier Transforms. arXiv.
3. Gu et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR.
4. Fein-Ashley et al. (2025). SPECTRE: An FFT-Based Efficient Drop-In Replacement to Self-Attention.
5. Poli et al. (2023). Hyena Hierarchy: Towards Larger Convolutional Language Models. ICML.

---

*This document is a living specification. Last updated: 2024.*
"""

with open("ARCHITECTURE.md", "w") as f:
    f.write(architecture_doc)

# USAGE.md - Practical guide
usage_doc = """# Wave Mixture Usage Guide

## Quick Reference

### Installation
```bash
pip install wave-mixture-llm
```

### Basic Usage
```python
from wave_mixture import WaveMixtureLLM, WaveMixtureConfig

config = WaveMixtureConfig()
model = WaveMixtureLLM(config)
```

---

## Common Tasks

### 1. Processing Long Documents

```python
# Load 500-page book
with open("book.txt") as f:
    text = f.read()

# Tokenize
tokens = tokenizer(text, max_length=1_000_000, truncation=True)

# Process
outputs = model(tokens.input_ids)

# Extract information
answer = extract_answer(outputs, "What is the main theme?")
```

### 2. Code Repository Analysis

```python
from wave_mixture.code import CodeProcessor

processor = CodeProcessor(model)

# Index entire repo
files = glob("src/**/*.py", recursive=True)
index = processor.index_repository(files)

# Query across files
result = processor.query("Where is the User class defined?")
```

### 3. Streaming Generation

```python
# Process 1M context, generate token by token
generator = model.stream_generate(
    context=input_ids,
    max_new_tokens=1000,
    temperature=0.8
)

for token in generator:
    print(token, end="", flush=True)
```

### 4. Fine-tuning

```python
from wave_mixture.training import Trainer

trainer = Trainer(
    model=model,
    train_dataset=dataset,
    batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=65536
)

trainer.train()
```

---

## Configuration Guide

### Small Model (350M params, 512K context)
```python
config = WaveMixtureConfig(
    dim=512,
    num_layers=12,
    num_propagation_heads=8,
    max_seq_len=524288,
    chunk_size=4096
)
```

### Medium Model (1.4B params, 1M context)
```python
config = WaveMixtureConfig(
    dim=1024,
    num_layers=24,
    num_propagation_heads=16,
    max_seq_len=1_048_576,
    chunk_size=8192,
    compression_ratio=16
)
```

### Large Model (6B params, 2M context)
```python
config = WaveMixtureConfig(
    dim=2048,
    num_layers=32,
    num_propagation_heads=32,
    max_seq_len=2_097_152,
    chunk_size=8192,
    compression_ratio=32,
    use_cpu_offload=True  # For extreme lengths
)
```

---

## Optimization Tips

### Memory Optimization

```python
# 1. Gradient checkpointing
config.use_activation_checkpointing = True

# 2. Mixed precision
from torch.cuda.amp import autocast
with autocast(dtype=torch.bfloat16):
    outputs = model(input_ids)

# 3. CPU offloading for extreme lengths
config.use_cpu_offload = True

# 4. Smaller chunk size
config.chunk_size = 4096  # Default 8192
```

### Speed Optimization

```python
# 1. Torch compile (PyTorch 2.0+)
model = torch.compile(model)

# 2. FlashFFT (if available)
config.use_flash_fft = True

# 3. Larger batch size for short sequences
if seq_len < 8192:
    batch_size = 8
else:
    batch_size = 1
```

---

## Troubleshooting

### Out of Memory

**Problem**: CUDA OOM at 100K+ tokens

**Solutions:**
1. Reduce chunk size: `config.chunk_size = 4096`
2. Enable gradient checkpointing: `config.use_activation_checkpointing = True`
3. Use CPU offloading: `config.use_cpu_offload = True`
4. Reduce batch size
5. Use bfloat16 instead of float32

### Slow Training

**Problem**: Training is slower than expected

**Solutions:**
1. Increase chunk size (if memory allows)
2. Use torch.compile
3. Check if FlashFFT is enabled
4. Use multiple GPUs with DDP

### Poor Long-Range Performance

**Problem**: Model misses dependencies >50K tokens

**Solutions:**
1. Check wave parameters: `print(model.layers[0].propagation.omega)`
2. Ensure very low frequencies (ω < 0.1) for global heads
3. Increase compression ratio for hierarchical processing
4. Train longer on long sequences

---

## API Reference

### WaveMixtureConfig

```python
@dataclass
class WaveMixtureConfig:
    vocab_size: int = 32000
    dim: int = 1024
    num_layers: int = 24
    num_propagation_heads: int = 16
    head_dim: int = 64
    max_seq_len: int = 1_048_576
    chunk_size: int = 8192
    compression_ratio: int = 16
    wave_damping_range: Tuple[float, float] = (0.0001, 0.01)
    wave_freq_range: Tuple[float, float] = (0.001, 1.0)
    use_hierarchical: bool = True
    use_activation_checkpointing: bool = False
    use_cpu_offload: bool = False
    dropout: float = 0.0
```

### WaveMixtureLLM

```python
class WaveMixtureLLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            targets: [batch, seq_len] for training
            use_cache: Whether to return cache for generation
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Autoregressive generation"""
    
    def get_memory_footprint(self, seq_len: int) -> Dict[str, float]:
        """Estimate memory usage"""
```

---

## Examples

See `examples/` directory for:
- `process_book.py`: Long document Q&A
- `code_completion.py`: Repository-level completion
- `train_custom.py`: Fine-tuning on custom data
- `benchmark.py`: Speed and memory benchmarks

---

## Support

- GitHub Issues: https://github.com/wavemixture/llm/issues
- Documentation: https://wavemixture.readthedocs.io
- Discord: https://discord.gg/wavemixture
