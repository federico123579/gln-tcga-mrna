// ============================================================================
// GLN-TCGA Binary Classification Report
// FIXME: title
// ============================================================================

#import "template.typ": *

#show: doc => template(
  authors: (
    ("Federico Lolli", "250835", "10720843"),
  ),
  instructors: (
    "Prof. Marcello Restelli",
  ),
  meta_title: "Meta title", // FIXME
  title: "Title", // FIXME
  subtitle: "Subtitle", // FIXME
  course_name: "Multidisciplinary Project (5 CFU)",
  academic_year: "2025-2026",
  abstract: [], // FIXME
  doc,
)

// ============================================================================
// INTRODUCTION
// ============================================================================

= Introduction

Gated Linear Networks (GLNs) are a class of neural networks introduced by @gln that replace global backpropagation with local convex learning and a contextual gating mechanism. Despite their theoretical appeal (online learning, modularity, and a natural notion of “active-path” explanations), available implementations are often hard to reproduce due to outdated dependencies and vague experimental settings.

This work provides:

+ A clean, modular GLN implementation in PyTorch #footnote([de-facto standard for the deep learning research community, see @pytorch]) designed for extension and controlled experimentation, following best practices for reproducible research and GPU acceleration.
+ An empirical study on a real high-dimensional biomedical task (TCGA-BRCA mRNA-based tumor vs. normal classification #short-cite(<tcga-brca>)).

Beyond predictive performance, an analysis of interpretability is performed using Saliency attribution and Integrated Gradients #short-cite(<integrated-gradients>) methods, revealing important constraints when applying explainability techniques to GLNs.

== Project Objective
Implement the GLN architecture from first principles in PyTorch (including geometric mixing and contextual gating), with emphasis on a reproducible and extensible codebase, and evaluate it on a real-world high-dimensional genomic classification task (TCGA-BRCA tumor vs. normal), focusing on both predictive performance and interpretability.

== Motivation
Two needs motivate this project. First, GLNs are typically presented as a practically attractive alternative to deep networks because each unit optimizes a local convex objective, enabling efficient online learning without end-to-end gradient credit assignment. Second, genomic classification operates in an extreme $p gt.double n$ regime (tens of thousands of genes and relatively few samples), where reproducibility and interpretability are as important as raw accuracy. A modular, well-specified implementation is required to assess whether GLNs' theoretical properties translate into reliable gains on real biomedical data.

== Problem Statement
Given TCGA Breast Invasive Carcinoma (BRCA) transcriptomic profiles with ~20k mRNA expression features per sample, learn a binary classifier that distinguishes tumor from normal tissue. The core challenges are (i) very high input dimensionality, (ii) class imbalance, and (iii) the requirement to understand which biological features drive decisions rather than only achieving high test accuracy.

== Report Structure
The report is split into two parts:
- *Part I* introduces the theory behind GLNs (local learning, geometric mixing, and half-space contextual gating) and describes the PyTorch implementation choices that enable reproducible experiments and architectural variations.
- *Part II* presents the TCGA-BRCA experimental setup and results, then focuses on interpretability: attribution with Integrated Gradients and Saliency attribution methods, followed by a discussion of the observed accuracy-interpretability tension in high-dimensional genomics.

#pagebreak()

// ============================================================================
// PART I
// ============================================================================

#part-section(
  [Gated Linear Networks #sym.dash.em Theory and Implementation],
  "Understanding the GLN architecture and building it from first principles.",
)

= Theoretical Foundation

Gated Linear Networks (GLNs) combine two ideas from @gln: (i) *geometric mixing*, which turns a linear combination of logits into a probability while preserving convexity of the log-loss in the weights, and (ii) *contextual gating*, which selects among multiple weight vectors based on side information (typically a random half-space partition of an input-dependent context). Stacking such units yields a deep architecture that can be trained online with local updates, without end-to-end backpropagation.

== From Backpropagation to Local Learning
In a standard neural network, parameters are optimized by minimizing a global objective and propagating gradients through the entire computation graph. This couples layers tightly: an upstream change modifies downstream representations, so the “credit assignment” problem is inherently global and the loss landscape is generally non-convex.

GLNs instead attach a *local* loss to each unit (log-loss for Bernoulli outputs). A unit treats its inputs (probabilities produced by the previous layer) as fixed features and updates only its own mixing weights using Online Gradient Descent (OGD). Under the geometric-mixing parameterization, the per-unit loss is convex in the unit’s weights, giving a stronger optimization story than typical deep networks: each unit tracks the best predictor available within its gated linear class, even in a streaming setting @gln.

Practically, this means training can be performed in a single pass over the data, with updates applied immediately after each example (or small batch), making GLNs naturally suited to online/continual learning scenarios.

== The Geometric Mixing Formula
Each GLN neuron operates on *probability inputs* $bold(p) in (0,1)^d$. Let
$ "logit"(p) = ln(p / (1 - p)) $
and $sigma(z) = 1 / (1 + exp(-z))$.

Given weights $bold(w) in RR^d$, the neuron outputs
$ hat(p) = sigma(bold(w) dot "logit"(bold(p))). $

Equivalently, this is a normalized product-of-experts form:
$ hat(p) = (product_i p_i^(w_i)) / (product_i p_i^(w_i) + product_i (1 - p_i)^(w_i)). $

Two consequences from @gln are central for learning:
- With log-loss $ell(y, hat(p)) = -y ln hat(p) - (1-y) ln(1-hat(p))$, the loss is *convex* in $bold(w)$.
- The gradient has a simple form proportional to the logit inputs, enabling stable OGD updates.

For numerical stability, implementations clamp inputs to $[epsilon, 1-epsilon]$ before applying `logit`, and apply a projection/clamp step $bold(w) <- "clip"(bold(w), -B, B)$ after each update (the bounded-domain assumption used by OGD analyses in @gln).

== Contextual Gating (Half-Space Partitioning)
Geometric mixing alone is still a generalized linear model over logit-features. GLNs gain capacity by allowing *multiple* weight vectors per neuron and selecting among them via a context function.

A common context mechanism in @gln is a set of $K$ random half-spaces over side information $bold(z)$ (often $bold(z)$ is the original input or a fixed transform of it). For $k in {1..K}$:
$ h_k(bold(z)) = [ bold(v)_k dot bold(z) + b_k > 0 ], $
with $(bold(v)_k, b_k)$ sampled once at initialization and kept fixed.

The resulting binary vector $bold(h) in {0,1}^K$ indexes one of $2^K$ regions; each region has its own weight vector $bold(w)_{bold(h)}$. Intuitively:
- within a region, the neuron behaves like a convex, well-optimized linear predictor (in logit space);
- globally, the neuron is piecewise-defined across regions, yielding a non-linear model without learning the gating boundaries.

This “fixed random partition + learned experts” design is a key reason GLNs can be trained with purely local convex updates while still achieving non-linear behavior @gln.

== Why GLNs Are Special
GLNs are interesting less for raw expressivity than for the combination of *capacity* and *training dynamics*:
- *Online learning by design*: local OGD updates allow effective single-pass training and fast adaptation.
- *Reduced interference*: because each context indexes its own weights, updates for one region minimally affect others (helpful under non-stationarity).
- *A structured explanation primitive*: for a given input, the active context picks a specific set of weights, so the prediction can be “collapsed” to an input-dependent linear form (useful for auditing, though not guaranteed to map cleanly to semantically meaningful features in high-dimensional tabular settings).
- *Theoretical tractability*: convex per-unit losses support convergence/regret guarantees under standard OGD assumptions @gln.

== The GLN Architecture
The GLN used in this work follows the standard blueprint from @gln:
- *Input-to-probability mapping*: continuous features are transformed to $(0,1)$ (e.g., via normalization + sigmoid + clipping) so they can be interpreted as Bernoulli probabilities for geometric mixing.
- *Stacked gated layers*: each layer contains multiple neurons; each neuron receives probability inputs from the previous layer and selects a context using side information (often the original input vector, so all layers gate on a common reference).
- *Bias feature*: a constant probability input is appended to each layer to allow non-zero intercepts under mixing.
- *Output Bernoulli predictor*: the final layer produces $hat(p)(y=1|x)$.

Training proceeds online: for each example, compute layer outputs, then update each layer’s weights using its local log-loss (optionally aggregating per-neuron losses within a layer). Importantly, the update rule treats lower-layer outputs as fixed inputs for the layer being updated, matching the “no cross-layer credit assignment” principle of GLNs @gln.

= PyTorch Implementation

== Module Structure
The codebase mirrors the conceptual decomposition of @gln:
- *Core math utilities*: stable `logit`, `sigmoid`, clamping, and geometric-mix forward pass.
- *Layer modules*: a base input-to-probability layer, gated geometric-mixing layers, and an output Bernoulli layer.
- *Model wrapper*: a `GLN` module that composes layers, controls context source, and exposes `forward` and training helpers.

This separation makes it easy to swap architectural choices (number of layers/neurons, context dimensionality, context source) while keeping the learning rule unchanged.

== Key Implementation Details
A few implementation details are critical to match the intended GLN behavior:
- *Tensor shapes*: for `out_dim` neurons, `in_dim` inputs, and `K` context bits, store weights as
  `W: [out_dim, 2^K, in_dim]`
  so each neuron has an independent expert table over contexts.
- *Context indexing*: compute the $K$ half-space bits per sample, pack them into an integer index in `[0, 2^K)`, and gather the corresponding slice of `W` efficiently (vectorized over batch and neurons).
- *Numerical stability*: clamp probabilities to `[eps, 1-eps]` before `logit`; optionally clamp logits to a finite range; project weights to `[-B, B]` after each optimizer step.
- *Local-loss training*: compute a Bernoulli log-loss per layer (or per neuron) using that layer’s prediction and the target; update only that layer’s parameters. In PyTorch, this is typically implemented by stopping gradient flow *between* layers for the purpose of weight updates (e.g., using detached inputs to each layer during that layer’s loss computation), reflecting the GLN local-learning assumption in @gln.

== Design Choices for Flexibility
The implementation is designed to support controlled experiments:
- *Context source as a parameter*: gate on raw input, normalized input, or previous-layer activations to test how partitions interact with domain geometry.
- *Deterministic random hyperplanes*: initialize context hyperplanes with a seeded generator and store them as non-trainable buffers so checkpoints fully capture the partitioning.
- *Device-agnostic execution*: keep buffers and parameters on the same device, allowing CPU/GPU/MPS runs without code changes.
- *Experiment hooks*: expose per-layer losses/accuracies and context-usage statistics (e.g., how many contexts are visited), which are often informative when diagnosing training/interpretability behavior.

#pagebreak()

// ============================================================================
// PART II
// ============================================================================

#part-section(
  "Application to Genomic Data",
  "Applying GLN to real cancer data and investigating what the model learns.",
)

= The TCGA-BRCA Dataset

// FIXME

== Data Source and Acquisition // FIXME
== Dataset Characteristics // FIXME

= Experimental Results

// FIXME

== Hyperparameter Configurations Tested // FIXME
== Online Learning Behavior // FIXME
== Baseline Comparisons // FIXME
== Cross-Validation Results // FIXME

= Interpretability Analysis

// FIXME

== The Promise of Interpretability // FIXME
== Integrated Gradients Method // FIXME
== Expected vs. Observed Results // FIXME
== Architectural Modifications Tested // FIXME
== Root Cause Analysis // FIXME
=== Primary Cause: Random Fixed Hyperplanes // FIXME
=== Secondary Cause: Sigmoid Saturation // FIXME
== Alternative Attribution: Saliency // FIXME

= Discussion

// FIXME

== The Interpretability Paradox // FIXME
== When GLN Interpretability Works vs. Fails // FIXME
== Practical Implications // FIXME

= Conclusion

// FIXME

== Summary of Contributions // FIXME
== Future Work // FIXME

// ============================================================================
// REFERENCES & APPENDIX
// ============================================================================

// Bibliography handled by the template

// FIXME APPENDIX
