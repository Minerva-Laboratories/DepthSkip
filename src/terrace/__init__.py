"""Terrace: training-free layer skipping for the prompt pass of hybrid (DeltaNet + attention) LMs.

Prompt tokens stop at a cut layer; every layer above it writes only its cache, from a closed-form (ridge) map of the
cut's input. Depth can differ per token (graded windows: recent and first tokens deeper), and the upper layers can
instead keep their mixers and replace only MLPs and/or attention outputs (part skips).
"""
from .maps import (LAMS, CacheOnlyLayer, MLPSurrogates, SkipMaps, context_features, fit_mlp_surrogates, fit_skip_maps,
                   skipped_share, skipping)
from .depth import (DepthPlan, depth_cache, depth_skipping, error_score, plan_skipped, seg_lengths, segment_cache,
                    segment_prefill)
from .parts import PartSkip, fit_parts, part_share, part_skip
from .model import Model, cache_tensors, use_reference_kernels

__version__ = "0.1.0"
__all__ = ["LAMS", "CacheOnlyLayer", "MLPSurrogates", "SkipMaps", "context_features", "fit_mlp_surrogates",
           "fit_skip_maps", "skipped_share", "skipping", "DepthPlan", "depth_cache", "depth_skipping", "error_score",
           "plan_skipped", "seg_lengths", "segment_cache", "segment_prefill", "PartSkip", "fit_parts", "part_share",
           "part_skip", "Model", "cache_tensors", "use_reference_kernels"]
