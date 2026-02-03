"""
Captum-based attribution methods for GLN models.

Provides optimized Integrated Gradients and Permutation Importance
using Captum library, with correlation analysis between methods.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import gln
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import FeaturePermutation, IntegratedGradients
from scipy import stats
from tqdm import tqdm

from gln_tcga.plotting import (
    KNOWN_CANCER_GENES,
    plot_contribution_violin,
    plot_gate_usage_heatmap,
    plot_gene_contrib_vs_logit,
    plot_known_cancer_genes,
    plot_rank_correlation_highlight,
    plot_sample_saliency,
    plot_sample_waterfall,
    plot_top_genes,
)

if TYPE_CHECKING:
    pass


class GLNWithTransform(nn.Module):
    """Wrapper combining InputTransformer + GLN for Captum compatibility.

    Captum requires a single nn.Module that takes raw inputs and produces outputs.
    This wrapper combines the sigmoid transformation (InputTransformer) with the
    GLN model, allowing Captum to properly compute attributions through the full
    input space rather than the pre-transformed space.
    """

    def __init__(self, model: gln.GLN, transformer: gln.InputTransformer):
        """Initialize wrapper.

        Args:
            model: Trained GLN model.
            transformer: Fitted InputTransformer from training.
        """
        super().__init__()
        self.model = model
        self.transformer = transformer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer and model.

        Args:
            x: Raw input tensor (before transformation).

        Returns:
            Model predictions.
        """
        x_transformed = self.transformer.transform(x)
        return self.model(x_transformed)


def _collapse_active_weights(
    model: gln.GLN,
    context: torch.Tensor,
) -> torch.Tensor:
    """Collapse active GLN weight matrices into a per-sample effective weight.

    The resulting weight vectors correspond to the data-dependent linear
    coefficients in logit space for each input sample.

    Args:
        model: Trained GLN model.
        context: Transformed input used for gating (batch_size, input_size).

    Returns:
        Tensor of shape (batch_size, input_size_with_bias) containing effective weights.
    """
    weight_mats: list[torch.Tensor] = []
    for layer in model.layers:
        weight_mats.append(layer._slice_weights(context))
    weight_mats.append(model.out_layer._slice_weights(context))

    W_eff = weight_mats[0]
    for W in weight_mats[1:]:
        expected_in = W.shape[2]
        if W_eff.shape[1] < expected_in:
            pad = expected_in - W_eff.shape[1]
            zeros = torch.zeros(
                (W_eff.shape[0], pad, W_eff.shape[2]),
                device=W_eff.device,
                dtype=W_eff.dtype,
            )
            W_eff = torch.cat([W_eff, zeros], dim=1)
        elif W_eff.shape[1] > expected_in:
            raise ValueError(
                f"Collapsed weights shape mismatch: expected input size {expected_in} but got {W_eff.shape[1]}."
            )

        W_eff = torch.bmm(W, W_eff)

    return W_eff.squeeze(1)


def compute_gln_saliency(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    gene_names: list[str],
    *,
    batch_size: int = 64,
    drop_bias: bool = True,
    rank_by: str = "contrib",
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    """Compute GLN saliency weights via collapsed active weight matrices.

    This follows the GLN paper's interpretability claim: for a fixed input,
    the model is linear in logit space, so the collapsed weights provide a
    direct saliency map without gradient backpropagation.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        gene_names: List of gene names corresponding to input features.
        batch_size: Batch size for processing samples.
        drop_bias: Whether to drop the bias term from analysis.
        rank_by: Ranking metric: "contrib" (mean |contribution|) or "weight" (mean |weight|).

    Returns:
        Tuple of:
            - DataFrame with gene names, importance scores, and direction
            - Raw saliency weight tensor (n_samples, n_genes)
            - Raw contribution tensor (n_samples, n_genes)
    """
    model.eval()

    device = next(model.parameters()).device
    n_samples = X.shape[0]

    all_weights: list[torch.Tensor] = []
    all_contribs: list[torch.Tensor] = []

    batch_iter = range(0, n_samples, batch_size)
    batch_iter = tqdm(batch_iter, desc="Computing GLN saliency", unit="batch")

    for i in batch_iter:
        batch_X = X[i : i + batch_size]
        batch_X = batch_X.cpu()

        # Use transformer on CPU, then move to device for model weights
        batch_transformed = transformer.transform(batch_X).to(device)
        context = batch_transformed

        # Effective weights in logit space
        weights = _collapse_active_weights(model, context)

        # Compute input logit to obtain signed contribution
        base_out = model.base_layer(batch_transformed)
        logit_inputs = torch.logit(base_out, eps=model.eps)

        if drop_bias and model.base_layer.raw_b is not None:
            weights = weights[:, 1:]
            logit_inputs = logit_inputs[:, 1:]

        contributions = weights * logit_inputs

        all_weights.append(weights.detach().cpu())
        all_contribs.append(contributions.detach().cpu())

    weights = torch.cat(all_weights, dim=0)
    contributions = torch.cat(all_contribs, dim=0)

    mean_abs_weights = weights.abs().mean(dim=0).numpy()
    mean_abs_contribs = contributions.abs().mean(dim=0).numpy()
    mean_signed_contrib = contributions.mean(dim=0).numpy()

    if rank_by not in {"contrib", "weight"}:
        raise ValueError(f"rank_by must be 'contrib' or 'weight', got {rank_by!r}.")

    importance = mean_abs_contribs if rank_by == "contrib" else mean_abs_weights

    df = pd.DataFrame(
        {
            "gene": gene_names[: len(importance)],
            "importance": importance,
            "weight_importance": mean_abs_weights,
            "contrib_importance": mean_abs_contribs,
            "direction": np.sign(mean_signed_contrib),
            "signed_contrib_mean": mean_signed_contrib,
        }
    )

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["rank_by"] = rank_by

    return df, weights, contributions


def _summarize_contributions_by_class(
    contributions: torch.Tensor,
    y: torch.Tensor,
) -> pd.DataFrame:
    """Compute class-conditional contribution summaries.

    Args:
        contributions: Tensor of shape (n_samples, n_genes).
        y: Tensor of labels (n_samples,).

    Returns:
        DataFrame with per-gene mean(|contribution|) and mean(contribution)
        for each class, plus a delta column.
    """
    y_np = y.detach().cpu().numpy().astype(int)
    c = contributions.detach().cpu()

    out = {}
    for label in (0, 1):
        mask = torch.from_numpy(y_np == label)
        if int(mask.sum()) == 0:
            out[f"mean_abs_contrib_y{label}"] = torch.full((c.shape[1],), float("nan"))
            out[f"mean_signed_contrib_y{label}"] = torch.full(
                (c.shape[1],), float("nan")
            )
        else:
            c_lab = c[mask]
            out[f"mean_abs_contrib_y{label}"] = c_lab.abs().mean(dim=0)
            out[f"mean_signed_contrib_y{label}"] = c_lab.mean(dim=0)

    df = pd.DataFrame(
        {
            "mean_abs_contrib_y0": out["mean_abs_contrib_y0"].numpy(),
            "mean_abs_contrib_y1": out["mean_abs_contrib_y1"].numpy(),
            "mean_signed_contrib_y0": out["mean_signed_contrib_y0"].numpy(),
            "mean_signed_contrib_y1": out["mean_signed_contrib_y1"].numpy(),
        }
    )
    df["delta_abs_contrib_y1_minus_y0"] = (
        df["mean_abs_contrib_y1"] - df["mean_abs_contrib_y0"]
    )
    return df


def compute_gln_gate_usage(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Compute gate usage frequency per layer and class.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        y: Target labels.
        batch_size: Batch size for processing.

    Returns:
        DataFrame with columns: layer, class, gate_index, count, frequency.
    """
    model.eval()
    device = next(model.parameters()).device

    layers = list(model.layers) + [model.out_layer]
    layer_names = [f"layer_{i}" for i in range(len(model.layers))] + ["out_layer"]
    n_gates = 2**model.context_dimension

    counts: dict[str, dict[int, np.ndarray]] = {
        name: {
            0: np.zeros(n_gates, dtype=np.int64),
            1: np.zeros(n_gates, dtype=np.int64),
        }
        for name in layer_names
    }

    n_samples = X.shape[0]
    batch_iter = range(0, n_samples, batch_size)
    batch_iter = tqdm(batch_iter, desc="Computing gate usage", unit="batch")

    for i in batch_iter:
        batch_X = X[i : i + batch_size].cpu()
        batch_y = y[i : i + batch_size].detach().cpu().numpy().astype(int)

        context = transformer.transform(batch_X).to(device)

        for layer, name in zip(layers, layer_names):
            idx = layer._slicing_indexes(context).detach().cpu()
            for label in (0, 1):
                mask = batch_y == label
                if not np.any(mask):
                    continue
                values = idx[mask].reshape(-1).numpy()
                counts[name][label] += np.bincount(values, minlength=n_gates)

    rows = []
    for name in layer_names:
        for label in (0, 1):
            total = counts[name][label].sum()
            freq = counts[name][label] / total if total > 0 else np.zeros(n_gates)
            for gate_index in range(n_gates):
                rows.append(
                    {
                        "layer": name,
                        "class": label,
                        "gate_index": gate_index,
                        "count": int(counts[name][label][gate_index]),
                        "frequency": float(freq[gate_index]),
                    }
                )

    return pd.DataFrame(rows)


def compute_integrated_gradients(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    gene_names: list[str],
    *,
    y: torch.Tensor | None = None,
    baseline_type: str = "mean",
    n_steps: int = 50,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, torch.Tensor]:
    """Compute Integrated Gradients attribution using Captum.

    Uses Captum's optimized IntegratedGradients implementation for computing
    feature attributions. The GLNWithTransform wrapper ensures gradients flow
    through the full input transformation.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        gene_names: List of gene names corresponding to input features.
        baseline_type: Type of baseline to use:
            - "mean": Dataset mean (transforms to ~0.5 for all genes) - RECOMMENDED
            - "zero": Zero baseline in raw space
        n_steps: Number of steps for Riemann approximation.
        batch_size: Batch size for processing samples.

    Returns:
        Tuple of:
            - DataFrame with gene names, importance scores, and direction
            - Raw attribution tensor (n_samples, n_genes)
    """
    # Wrap model for Captum
    wrapped_model = GLNWithTransform(model, transformer)
    wrapped_model.eval()

    # Initialize Captum's IntegratedGradients
    ig = IntegratedGradients(wrapped_model)

    # Choose baseline based on type (RAW space, before transformation)
    if baseline_type == "mean":
        # Use fitted mean as baseline - transforms to ~0.5 for many genes
        baseline = transformer.means.unsqueeze(0).expand(X.shape[0], -1)
    elif baseline_type == "normal":
        if y is None:
            warnings.warn(
                "baseline_type='normal' requested but labels y were not provided; falling back to baseline_type='mean'.",
                stacklevel=2,
            )
            baseline = transformer.means.unsqueeze(0).expand(X.shape[0], -1)
        else:
            normal_mask = y == 0
            if int(normal_mask.sum()) > 0:
                baseline = (
                    X[normal_mask].mean(dim=0, keepdim=True).expand(X.shape[0], -1)
                )
            else:
                warnings.warn(
                    "baseline_type='normal' requested but no normal samples found (y==0); falling back to baseline_type='mean'.",
                    stacklevel=2,
                )
                baseline = transformer.means.unsqueeze(0).expand(X.shape[0], -1)
    elif baseline_type == "zero":
        baseline = torch.zeros_like(X)
    else:
        warnings.warn(
            f"Unknown baseline_type={baseline_type!r}; falling back to baseline_type='mean'.",
            stacklevel=2,
        )
        baseline = transformer.means.unsqueeze(0).expand(X.shape[0], -1)

    # Process in batches
    all_attributions = []
    n_samples = X.shape[0]

    batch_iter = range(0, n_samples, batch_size)
    batch_iter = tqdm(batch_iter, desc="Computing IG attributions", unit="batch")

    for i in batch_iter:
        batch_X = X[i : i + batch_size]
        batch_baseline = baseline[i : i + batch_size]

        # Compute attributions for this batch
        attributions = ig.attribute(
            batch_X,
            baselines=batch_baseline,
            n_steps=n_steps,
            internal_batch_size=min(batch_size, len(batch_X)),
        )
        all_attributions.append(attributions.detach())

    # Concatenate all batches
    attributions = torch.cat(all_attributions, dim=0)

    # Aggregate across samples: mean absolute attribution
    mean_abs_attr = attributions.abs().mean(dim=0).numpy()

    # Compute signed mean to see direction of effect
    mean_signed_attr = attributions.mean(dim=0).numpy()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names[: len(mean_abs_attr)],
            "importance": mean_abs_attr,
            "direction": np.sign(mean_signed_attr),  # +1 = tumor, -1 = normal
            "signed_importance": mean_signed_attr,
        }
    )

    # Sort by absolute importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df, attributions


def compute_permutation_importance(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    gene_names: list[str],
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Compute Permutation Importance using Captum.

    Model-agnostic feature importance that measures the decrease in model
    performance when a feature's values are randomly shuffled.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        y: Target labels.
        gene_names: List of gene names corresponding to input features.
        batch_size: Not used directly but kept for API consistency.

    Returns:
        DataFrame with gene names and importance scores.
    """
    # Wrap model for Captum
    wrapped_model = GLNWithTransform(model, transformer)
    wrapped_model.eval()

    # Initialize Captum's FeaturePermutation
    perm = FeaturePermutation(wrapped_model)

    print("Computing permutation importance...")
    print(f"  n_features: {X.shape[1]}")

    # Compute permutation importance
    # FeaturePermutation measures the change in output when permuting each feature
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributions = perm.attribute(
            X,
            perturbations_per_eval=2,
            n_perturb_samples=10,
            show_progress=True,
        )

    # Aggregate: mean absolute change per feature
    mean_importance = attributions.abs().mean(dim=0).numpy()

    # Create DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names[: len(mean_importance)],
            "importance": mean_importance,
        }
    )

    # Sort by importance descending
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


def compute_rank_correlation(
    ig_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute Spearman rank correlation between IG and Permutation rankings.

    Args:
        ig_df: DataFrame from compute_integrated_gradients with 'gene' and 'rank'.
        perm_df: DataFrame from compute_permutation_importance with 'gene' and 'rank'.

    Returns:
        Dictionary with keys:
            - spearman_r: Spearman correlation coefficient
            - p_value: Two-sided p-value
            - n_genes: Number of genes compared
    """
    # Merge on gene names to align rankings
    merged = ig_df[["gene", "rank"]].merge(
        perm_df[["gene", "rank"]],
        on="gene",
        suffixes=("_ig", "_perm"),
    )

    if merged.empty:
        return {
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "n_genes": 0,
        }

    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(merged["rank_ig"], merged["rank_perm"])

    return {
        "spearman_r": float(correlation),
        "p_value": float(p_value),
        "n_genes": len(merged),
    }


def compute_rank_correlation_generic(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, float]:
    """Compute Spearman rank correlation between two ranking DataFrames.

    Args:
        df_a: DataFrame with 'gene' and 'rank'.
        df_b: DataFrame with 'gene' and 'rank'.
        label_a: Label for df_a in the output.
        label_b: Label for df_b in the output.

    Returns:
        Dictionary with correlation metrics.
    """
    merged = df_a[["gene", "rank"]].merge(
        df_b[["gene", "rank"]],
        on="gene",
        suffixes=(f"_{label_a}", f"_{label_b}"),
    )

    if merged.empty:
        return {
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "n_genes": 0,
        }

    correlation, p_value = stats.spearmanr(
        merged[f"rank_{label_a}"],
        merged[f"rank_{label_b}"],
    )

    return {
        "spearman_r": float(correlation),
        "p_value": float(p_value),
        "n_genes": len(merged),
    }


def compute_top_rank_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    top_n: int = 100,
) -> dict[str, float]:
    """Compute overlap statistics between top-N ranked genes.

    Args:
        df_a: DataFrame with 'gene' and 'rank'.
        df_b: DataFrame with 'gene' and 'rank'.
        top_n: Number of top genes to compare.

    Returns:
        Dictionary with overlap size and Jaccard index.
    """
    top_a = set(df_a.sort_values("rank").head(top_n)["gene"])
    top_b = set(df_b.sort_values("rank").head(top_n)["gene"])

    if not top_a or not top_b:
        return {
            "top_n": top_n,
            "overlap": 0,
            "jaccard": float("nan"),
        }

    overlap = len(top_a.intersection(top_b))
    union = len(top_a.union(top_b))

    return {
        "top_n": top_n,
        "overlap": overlap,
        "jaccard": overlap / union if union > 0 else float("nan"),
    }


def generate_attributions_report(
    model: gln.GLN,
    transformer: gln.InputTransformer,
    X: torch.Tensor,
    y: torch.Tensor,
    gene_names: list[str],
    output_dir,
    *,
    method: str = "both",
    compute_correlation: bool = True,
    n_steps: int = 50,
    batch_size: int = 64,
    baseline_type: str = "mean",
    saliency_samples_per_class: int = 3,
) -> dict:
    """Generate comprehensive attribution report.

    Args:
        model: Trained GLN model.
        transformer: Fitted InputTransformer.
        X: Raw input tensor (before transformation).
        y: Target labels.
        gene_names: List of gene names corresponding to input features.
        output_dir: Directory to save outputs.
        method: Attribution method(s) to use: "ig", "permutation", "saliency",
            "both" (ig+permutation), or "all" (ig+permutation+saliency).
        compute_correlation: Whether to compute rank correlation (requires both methods).
        n_steps: Number of steps for IG.
        batch_size: Batch size for processing.
        baseline_type: Baseline type for IG.
        saliency_samples_per_class: Number of per-class sample saliency plots to generate.

    Returns:
        Dictionary containing results (DataFrames, correlation stats, etc.)
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if method == "both":
        methods = {"ig", "permutation"}
    elif method == "all":
        methods = {"ig", "permutation", "saliency"}
    else:
        methods = {method}

    # Compute Integrated Gradients
    if "ig" in methods:
        print("\n" + "-" * 40)
        print("Computing Integrated Gradients (Captum)...")
        print("-" * 40)

        ig_df, _ = compute_integrated_gradients(
            model,
            transformer,
            X,
            gene_names,
            y=y,
            baseline_type=baseline_type,
            n_steps=n_steps,
            batch_size=batch_size,
        )

        # Save to CSV
        ig_path = output_dir / "ig_importance.csv"
        ig_df.to_csv(ig_path, index=False)
        print(f"Saved IG importance to: {ig_path}")

        results["ig_df"] = ig_df

        # Print top genes
        print("\nTop 10 genes by IG:")
        for _, row in ig_df.head(10).iterrows():
            direction = "+" if row["direction"] > 0 else "-"
            print(
                f"  {row['rank']:3d}. {row['gene']:12s} ({direction}) importance={row['importance']:.4f}"
            )

        # Print known breast cancer gene ranks
        print("\n" + "-" * 40)
        print("Known Breast Cancer Genes Rankings:")
        print("-" * 40)
        known_rows = []
        for gene in KNOWN_CANCER_GENES:
            matches = ig_df[ig_df["gene"] == gene]
            if not matches.empty:
                row = matches.iloc[0]
                print(
                    f"  {gene:10s}: Rank {row['rank']:5d} (importance: {row['importance']:.4f})"
                )
                known_rows.append(
                    {
                        "gene": gene,
                        "rank": int(row["rank"]),
                        "importance": float(row["importance"]),
                        "direction": int(row["direction"]),
                        "signed_importance": float(row["signed_importance"]),
                    }
                )
            else:
                print(f"  {gene:10s}: NOT FOUND")
                known_rows.append(
                    {
                        "gene": gene,
                        "rank": None,
                        "importance": None,
                        "direction": None,
                        "signed_importance": None,
                    }
                )

        known_df = pd.DataFrame(known_rows)
        known_path = output_dir / "known_cancer_gene_ranks.csv"
        known_df.to_csv(known_path, index=False)
        print(f"Saved known cancer gene ranks to: {known_path}")

        # Generate plots
        print("\nGenerating plots...")
        plot_top_genes(
            ig_df,
            top_n=30,
            output_path=output_dir / "top_genes.png",
            title="Top 30 Genes by Integrated Gradients Attribution",
        )
        plot_known_cancer_genes(
            ig_df,
            output_path=output_dir / "known_cancer_genes.png",
        )

    # Compute Permutation Importance
    if "permutation" in methods:
        print("\n" + "-" * 40)
        print("Computing Permutation Importance (Captum)...")
        print("-" * 40)

        perm_df = compute_permutation_importance(
            model,
            transformer,
            X,
            y,
            gene_names,
            batch_size=batch_size,
        )

        # Save to CSV
        perm_path = output_dir / "permutation_importance.csv"
        perm_df.to_csv(perm_path, index=False)
        print(f"Saved permutation importance to: {perm_path}")

        results["perm_df"] = perm_df

        # Print top genes
        print("\nTop 10 genes by Permutation Importance:")
        for _, row in perm_df.head(10).iterrows():
            print(
                f"  {row['rank']:3d}. {row['gene']:12s} importance={row['importance']:.4f}"
            )

    # Compute correlation between methods
    if compute_correlation and {"ig", "permutation"}.issubset(methods):
        print("\n" + "-" * 40)
        print("Computing Rank Correlation...")
        print("-" * 40)

        correlation = compute_rank_correlation(results["ig_df"], results["perm_df"])

        # Save correlation results
        corr_path = output_dir / "rank_correlation.json"
        with open(corr_path, "w") as f:
            json.dump(correlation, f, indent=2)
        print(f"Saved correlation results to: {corr_path}")

        results["correlation"] = correlation

        print(f"\nSpearman correlation: r={correlation['spearman_r']:.4f}")
        print(f"P-value: {correlation['p_value']:.2e}")
        print(f"N genes: {correlation['n_genes']}")

    # Compute GLN saliency maps
    if "saliency" in methods:
        print("\n" + "-" * 40)
        print("Computing GLN Saliency Maps (collapsed weights)...")
        print("-" * 40)

        saliency_df, saliency_weights, saliency_contribs = compute_gln_saliency(
            model,
            transformer,
            X,
            gene_names,
            batch_size=batch_size,
            rank_by="contrib",
        )

        saliency_path = output_dir / "saliency_importance.csv"
        saliency_df.to_csv(saliency_path, index=False)
        print(f"Saved saliency importance to: {saliency_path}")

        results["saliency_df"] = saliency_df

        by_class = _summarize_contributions_by_class(saliency_contribs, y)
        by_class.insert(0, "gene", gene_names[: len(by_class)])
        by_class_path = output_dir / "saliency_contribs_by_class.csv"
        by_class.to_csv(by_class_path, index=False)
        print(f"Saved class-conditional saliency contributions to: {by_class_path}")

        merged_sal = saliency_df.merge(by_class, on="gene", how="left")
        merged_path = output_dir / "saliency_importance_with_class.csv"
        merged_sal.to_csv(merged_path, index=False)
        print(f"Saved merged saliency table to: {merged_path}")

        print("\nTop 10 genes by GLN Saliency (ranked by mean |contribution|):")
        for _, row in saliency_df.head(10).iterrows():
            direction = "+" if row["direction"] > 0 else "-"
            print(
                f"  {row['rank']:3d}. {row['gene']:12s} ({direction}) "
                f"contrib={row['contrib_importance']:.4f}  weight={row['weight_importance']:.4f}"
            )

        print("\nGenerating saliency plots...")
        plot_top_genes(
            saliency_df,
            top_n=30,
            output_path=output_dir / "top_genes_saliency.png",
            title="Top 30 Genes by GLN Saliency (mean |contribution|)",
        )
        plot_known_cancer_genes(
            saliency_df,
            output_path=output_dir / "known_cancer_genes_saliency.png",
            title="Known Breast Cancer Genes - GLN Saliency (mean |contribution|)",
        )

        top_scatter_genes = saliency_df["gene"].head(6).tolist()
        top_violin_genes = saliency_df["gene"].head(10).tolist()

        x_cpu = X.detach().cpu()
        logit_inputs = (x_cpu - transformer.means.cpu()) / transformer.stds.cpu()
        contribs_np = saliency_contribs.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy().astype(int)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        scatter_dir = output_dir / "saliency_scatter"
        for gene in top_scatter_genes:
            idx = gene_to_idx.get(gene)
            if idx is None:
                continue
            plot_gene_contrib_vs_logit(
                logit_inputs[:, idx].numpy(),
                contribs_np[:, idx],
                y_np,
                gene=gene,
                output_path=scatter_dir / f"contrib_vs_logit_{gene}.png",
                title=f"Contribution vs Expression: {gene}",
            )

        plot_contribution_violin(
            contribs_np,
            y_np,
            gene_names,
            top_genes=top_violin_genes,
            output_path=output_dir / "saliency_contrib_violin.png",
            title="Class-conditional Contribution Distributions (Top genes)",
        )

        # Sample-level saliency maps (one-vs-all in logit space)
        if saliency_samples_per_class > 0:
            print("\nGenerating sample-level saliency maps...")
            device = next(model.parameters()).device

            probs = []
            with torch.no_grad():
                for i in range(0, X.shape[0], batch_size):
                    batch_X = X[i : i + batch_size]
                    batch_transformed = transformer.transform(batch_X.cpu()).to(device)
                    batch_probs = model(batch_transformed).squeeze(1).detach().cpu()
                    probs.append(batch_probs)
            probs = torch.cat(probs, dim=0)

            y_np = y.detach().cpu().numpy().astype(int)
            probs_np = probs.numpy()

            sample_indices: list[int] = []
            for label in (0, 1):
                label_mask = y_np == label
                if not label_mask.any():
                    continue
                scores = probs_np if label == 1 else 1.0 - probs_np
                label_indices = np.where(label_mask)[0]
                label_scores = scores[label_indices]
                top_k = min(saliency_samples_per_class, len(label_indices))
                top_idx = label_indices[np.argsort(label_scores)[-top_k:][::-1]]
                sample_indices.extend(top_idx.tolist())

            for idx in sample_indices:
                weights = saliency_contribs[idx].numpy()
                pred_prob = probs_np[idx]
                label = int(y_np[idx])
                title = (
                    f"Sample Saliency (contribution) (idx={idx}, label={label}, "
                    f"p(tumor)={pred_prob:.3f})"
                )
                plot_sample_saliency(
                    gene_names,
                    weights,
                    output_path=output_dir / f"saliency_sample_{idx}_label_{label}.png",
                    top_n=30,
                    title=title,
                )

                plot_sample_waterfall(
                    weights,
                    gene_names,
                    output_path=output_dir
                    / f"saliency_sample_{idx}_label_{label}_waterfall.png",
                    title=title,
                    top_n_pos=15,
                    top_n_neg=15,
                )

                top_df = pd.DataFrame(
                    {
                        "gene": gene_names[: len(weights)],
                        "contribution": weights,
                        "abs_contribution": np.abs(weights),
                    }
                ).sort_values("abs_contribution", ascending=False)
                top_df.head(200).to_csv(
                    output_dir / f"saliency_sample_{idx}_top_genes.csv",
                    index=False,
                )

        results["saliency_weights"] = saliency_weights
        results["saliency_contribs"] = saliency_contribs

        gate_df = compute_gln_gate_usage(
            model,
            transformer,
            X,
            y,
            batch_size=batch_size,
        )
        gate_path = output_dir / "gate_usage.csv"
        gate_df.to_csv(gate_path, index=False)
        print(f"Saved gate usage stats to: {gate_path}")

        plot_gate_usage_heatmap(
            gate_df,
            output_dir=output_dir / "gate_usage",
            title_prefix="Gate Usage",
        )

    # Correlation between IG and saliency
    if compute_correlation and {"ig", "saliency"}.issubset(methods):
        print("\n" + "-" * 40)
        print("Computing IG vs Saliency Rank Correlation...")
        print("-" * 40)

        corr = compute_rank_correlation_generic(
            results["ig_df"],
            results["saliency_df"],
            label_a="ig",
            label_b="saliency",
        )
        overlap = compute_top_rank_overlap(
            results["ig_df"],
            results["saliency_df"],
            top_n=100,
        )
        saliency_corr = {**corr, **overlap}

        corr_path = output_dir / "saliency_rank_correlation.json"
        with open(corr_path, "w") as f:
            json.dump(saliency_corr, f, indent=2)
        print(f"Saved saliency correlation results to: {corr_path}")

        results["saliency_correlation"] = saliency_corr

        print(f"\nSpearman correlation: r={corr['spearman_r']:.4f}")
        print(f"P-value: {corr['p_value']:.2e}")
        print(
            f"Top-100 overlap: {overlap['overlap']} (Jaccard={overlap['jaccard']:.3f})"
        )

        plot_rank_correlation_highlight(
            results["ig_df"],
            results["saliency_df"],
            label_a="IG",
            label_b="GLN Saliency",
            known_genes=KNOWN_CANCER_GENES,
            output_path=output_dir / "rank_correlation_saliency_highlight.png",
            title="IG vs GLN Saliency (highlighted)",
        )

    return results
