"""
substrate.backend — MLX execution layer.

Where compiled PlanBundles meet real tensors. The IR and planner are
architecture-agnostic; this package is where Qwen2-shaped reality lives
in v0.1.

Public surface:
    MLXOpKernel    — Per-op execution against MLX modules (Qwen2-shaped).
    MLXForwardSession — End-to-end forward pass with KV caches, embeddings,
                         and lm_head. Wraps MLXOpKernel for single prompts.

Both classes are imported lazily by their consumers; this package's
__init__ does not import mlx so substrate stays installable on non-Apple
machines for IR/planner work.
"""

# Intentionally empty: imports happen at the consumer level.
