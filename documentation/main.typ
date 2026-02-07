// ============================================================================
// GLN-TCGA Binary Classification Report
// FIXME: title
// ============================================================================

#import "template.typ": *
#let data = yaml("data.yml")

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

Gated Linear Networks (GLNs) are a class of neural networks introduced by #long-cite(<gln>) that replace global backpropagation with local convex learning and a contextual gating mechanism. Despite their theoretical appeal (online learning, modularity, and a natural notion of “active-path” explanations), available implementations are often hard to reproduce due to outdated dependencies and vague experimental settings.

This work provides:

+ A clean, modular GLN implementation in PyTorch #footnote([de-facto standard for the deep learning research community, see #long-cite(<pytorch>)]) designed for extension and controlled experimentation, following best practices for reproducible research and GPU acceleration.
+ An empirical study on a real high-dimensional biomedical task (TCGA-BRCA mRNA-based tumor vs. normal classification @tcga-brca)).

Beyond predictive performance, an analysis of interpretability is performed using Saliency attribution and Integrated Gradients @integrated-gradients methods, revealing important constraints when applying explainability techniques to GLNs.

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

Gated Linear Networks (GLNs) are fundamentally based on two different learning mechanics @gln:

+ *geometric mixing*: a logit-space mixture that maps probability features back to a probability while keeping the Bernoulli log-loss convex in the mixing weights
+ *contextual gating*: a discrete selection mechanism that chooses among multiple expert weight vectors based on side information.

Stacking such gated geometric-mixing units results in a deep architecture that admits an online training rule with purely local updates and strong theoretical guarantees.

== Probability Features and Stable Logits <transform-input>
A key architectural constraint in @gln is that each layer consumes _probability features_ rather than unconstrained real-valued activations. In the paper's formulation, these initial probabilities are provided by a _base model_ (or _base transformation_) that maps the raw input into $(0,1)$; the GLN then composes gated geometric-mixing units on top of this probabilistic interface.

In this work, following the practical choice suggested by DeepMind's reference implementation @deepmind-repo, the base transform is a feature-wise sigmoid. To make this mapping invariant to affine shifts in location and scale, the raw vector $bold(x) in RR^p$ is first standardized using training-set statistics:
$ bold(x)' = (bold(x) - bold(mu)) / bold(s), $
where $bold(mu)$ and $bold(s)$ are per-feature mean and standard deviation estimated on the training set. The base probabilities are then obtained as
$ bold(p) = "clip"( sigma(bold(x)'), epsilon, 1 - epsilon ) in (0,1)^p, $
with $sigma$ the logistic sigmoid and $epsilon$ a small constant to prevent numerical issues at the boundaries. This stage provides the bounded probabilistic features required by the GLN units and reduces immediate sigmoid saturation due to arbitrary input scales.

The network then operates on $bold(p)$ through the stable logit transform
$ sigma^(-1)_epsilon (p) = "logit"( "clip"(p, epsilon, 1 - epsilon) ) = ln(p / (1 - p)). $

== Geometric Mixing
Let $bold(p) in (0,1)^d$ be the vector of probability features for a neuron (after optional bias concatenation), and let $bold(w) in RR^d$ be its mixing weights. The geometric-mixing prediction is
$ hat(p) = sigma(bold(w)^top sigma^(-1)_epsilon (bold(p))). $

The same prediction can be written more intuitively as a product-of-experts by moving to odds space. Define the odds of each input feature as $o_i$ and the combined odds as $o$:
$
  o_i := p_i / (1 - p_i), #h(2cm) o := product_i o_i^(w_i)
$

Converting back to a probability gives
$
  hat(p) = o / (1 + o)
  = (product_i p_i^(w_i)) / (product_i p_i^(w_i) + product_i (1 - p_i)^(w_i)).
$

Given Bernoulli target $y in {0,1}$, the local log-loss is
$ ell(y, hat(p)) = -y ln hat(p) - (1-y) ln(1-hat(p)). $
Under geometric mixing, $ell$ is convex in $bold(w)$ @gln. Moreover, the gradient has a particularly simple closed form:
$ nabla_w ell = (hat(p) - y) sigma^(-1)_epsilon (bold(p)). $
This is the algebraic reason GLNs admit stable, layer-local Online Gradient Descent (OGD) updates without backpropagation.

== Contextual Gating via Random Half-Spaces
Geometric mixing alone is a generalized linear model in logit space. Capacity is increased by equipping each neuron with a *table* of expert weights indexed by a discrete context.

In the implemented GLN, each neuron $j$ samples $K$ random half-spaces over a context vector $bold(z) in RR^q$:
$ h_{j,k}(bold(z)) = [ bold(v)_{j,k}^top bold(z) >= b_{j,k} ], quad k in {0,1,...,K-1}, $
where $bold(v)_{j,k}$ is sampled from a standard normal distribution and normalized to unit length, and $b_{j,k}$ is sampled from a standard normal distribution. The resulting bit-vector $bold(h)_j in {0,1}^K$ is packed into an integer index
$ c_j(bold(z)) = sum_{k=0}^{K-1} 2^k h_{j,k}(bold(z)) in {0,...,2^K-1}. $

Each neuron therefore stores a weight tensor
$ W_j in RR^(2^K times d), quad (W_j)_{c,:} = bold(w)_{j,c}, $
and uses $bold(w)_{j,c_j(bold(z))}$ as its active expert on input $bold(z)$. In this work, the context is taken to be the input itself (transformed as seen in @transform-input), and it is treated as fixed side information during local updates.

== Bias as a Probability Feature
To provide an intercept term while preserving the probability-feature interface, the implementation uses a learnable _bias probability_ $b in (0,1)$ appended as an additional input coordinate (and similarly appended between hidden layers). The bias is parameterized by an unconstrained scalar $r in RR$ via $b = sigma(r)$ and then concatenated to the feature vector, yielding an always-valid Bernoulli feature.

= PyTorch Implementation

The implementation, provided alongside this document, closely follows the decomposition suggested by #long-cite(<gln>), while exposing two complementary training modes (@training-regimes).

== Tensorization and Indexing
For a layer with `size` neurons, input dimension $d$, and context dimension $K$, the layer stores
$ W in RR^("size" times 2^K times d) $
Given a batch of contexts, the half-space tests produce a boolean tensor of shape $("size", "batch", K)$, which is packed into indices $c in {0,...,2^K-1}$ per neuron and sample. The corresponding expert vectors $W[j, c] in RR^d$ are gathered efficiently to compute the forward pass in a fully vectorized manner.

== Two Training Regimes <training-regimes>
The codebase supports two distinct ways of fitting the same architecture.

=== (A) Paper-faithful Online OGD (Local Learning)
In the online regime, each layer performs local OGD updates on its active expert weights, treating the inputs from the previous layer as fixed probability features. For a neuron with active expert weights $bold(w)_t$ at time $t$, local feature vector $bold(p)_t$ and learning rate $eta_t$, the update implemented is
$ bold(w)_{t+1} = Pi_W ( bold(w)_t - eta_t (hat(p)_t - y_t) sigma^(-1)_epsilon (bold(p)_t) ), $
where $Pi_W$ is the Euclidean projection onto a compact convex set $W$.

In this implementation, $W$ is chosen as a coordinate-wise hypercube $[w_min, w_max]^d$ (defaulting to $[0,1]^d$), so projection is an inexpensive clamp operation. The learning rate schedule is configurable, and includes the paper-standard decay
$ eta_t = eta_0 / sqrt(t), $
which yields an $O(sqrt(T))$ regret bound for OGD under standard assumptions (convex loss, bounded gradients, bounded domain). In practice, the implementation also provides a vectorized variant of the update: within a batch, gradients corresponding to samples that activate the same (neuron, context) expert are aggregated before applying the step, trading exact per-sample updates for speed.

=== (B) Batch Adam (End-to-end Optimization for Ablations)
For empirical comparisons and controlled experiments, the same GLN can be trained in a batch regime using standard end-to-end optimization. In this mode, the empirical risk is minimized:
$ min_theta (1/n) sum_{i=1}^n ell(y_i, f_theta(bold(p)_i)), $
where $theta$ collects all learnable parameters (in particular, the expert weight tables and bias parameters) and $f_theta$ is the compositional forward map of the stacked gated layers.

Optimization is performed with Adam in the experiments reported in this work, but the implementation is agnostic to the specific choice of optimizer: any PyTorch-compatible optimizer (e.g., SGD, AdamW, RMSprop, Adagrad) can be plugged in to update the expert weight tables and bias parameters.
Since contextual gating is discrete, the resulting objective is piecewise smooth: for a fixed set of active experts, gradients flow to the selected weights, while the gating boundaries remain fixed (hyperplanes are not trained). To stabilize training across optimizers, weights are optionally clamped to a finite range after each optimizer step.

The two regimes should be interpreted as complementary: online OGD matches the theoretical GLN learning rule and supports regret-style guarantees, whereas batch Adam intentionally relaxes the local-learning principle to enable ablations and stress-tests in the high-dimensional settings.

== Accelerator Compatibility and Vectorized Execution
All computations are expressed as standard PyTorch tensor operations and run unchanged on any backend supported by PyTorch's `device` interface (CPU, CUDA, MPS). Inputs, weight tables, gating hyperplanes, and intermediate tensors are created on the selected device, and both the forward pass (gating, expert selection, geometric mixing) and the optional batched OGD update are implemented without Python-level loops, relying instead on batched indexing/gather and fused elementwise ops.

== Reproducibility Benchmark (MNIST)
Results on the MNIST dataset @mnist are provided as an external reproducibility benchmark. The dataset is small enough to enable quick end-to-end runs and structured to evaluate (i) multiclass training, (ii) non-trivial generalization after short training, and (iii) interpretable saliency patterns consistent with digit strokes.

The benchmark suite lives in `gated-linear-networks/benchmarks`. The most reliable reproduction instructions (installation, CLI options, and output paths) are given in the benchmark README in that directory.

The benchmark trains a `MulticlassGLN` via a one-vs-all decomposition. Two training modes are available: paper-faithful online OGD (default) and batch backpropagation. The run summarized here uses online OGD and then generates a 2$times$5 saliency grid over the ten digit classes (see @mnist_saliency). The hyperparameter configuration and the per-class test accuracies are summarized in the tables below.

#let params = (
  (epochs, 1),
  (batchsize, 1),
  (lr, 0.01),
  ([#layer1, #layer2], [50, 25]),
  (ctxdim, 6),
  (lrsched, "sqrt"),
)
#let parameter_figure = [
  #figure(
    parameter_table((params)),
    caption: [MNIST benchmark hyperparameters used for the run.],
  ) <mnist_accuracies>
]

#let accuracy_figure = figure(
  image("assets/accuracies_barplot.png", width: 100%),
  caption: [Barplot of the MNIST per-class test accuracies],
)

#align(center, block(
  width: 100%,
  grid(
    columns: (60%, auto),
    align: horizon,
    gutter: 1cm,
    accuracy_figure, parameter_figure,
  ),
))

On this configuration, the overall test accuracy was #strong([92.03%]). Per-class accuracies are reported in @mnist_accuracies; the most challenging classes in this run were 5 and 9, while 1 achieved the highest accuracy.

#figure(
  image("assets/mnist_saliency_maps.png", width: 100%),
  caption: [MNIST signed saliency maps per digit class computed from the GLN's collapsed active weights. The patterns are qualitatively consistent with reference visualizations (salient strokes align with digit morphology), but are not expected to match exactly across runs due to random half-space partitions, seed sensitivity, and small differences in preprocessing and learning-rate schedules.],
) <mnist_saliency>

#pagebreak()

// ============================================================================
// PART II
// ============================================================================

#part-section(
  "Application to Genomic Data",
  "Applying GLN to real cancer data and investigating what the model learns.",
)

= The TCGA-BRCA Dataset

This work uses TCGA Breast Invasive Carcinoma (BRCA) data and frames it as a binary classification problem: given a vector of mRNA expression measurements, predict whether a tissue sample is *tumor* or *normal*. At a high level, each sample is represented by a high-dimensional feature vector (one feature per gene), which matches the setting described in the TCGA BRCA reference study @tcga-brca. Concretely, the files used in this project are obtained through the cBioPortal ecosystem @cbioportal.

== Data Source and Acquisition
Data are retrieved from the public cBioPortal DataHub repository @cbioportal-datahub, specifically from the curated study `brca_tcga_pan_can_atlas_2018`. To make acquisition reproducible and lightweight, the loader downloads only the two required expression matrices directly via HTTP from GitHub (rather than cloning the full repository):

- tumor samples: `data_mrna_seq_v2_rsem.txt`
- normal samples: `normals/data_mrna_seq_v2_rsem_normal_samples.txt`

Each file is a tab-separated _expression matrix_, whose rows index genes and whose columns index samples. Each entry $e_(g,s)$ represents the (RSEM-based) expression estimate of gene $g$ in sample $s$ @rsem. Downloaded artifacts are cached locally to make reruns deterministic and to avoid repeated network transfers.

== Dataset Construction and Cleaning
The loader parses the two tab-separated tables, removes metadata columns, and transposes the matrices so that each row is a sample and each column is a gene.

Tumor and normal files do not always contain exactly the same gene list. Therefore, the final feature set keeps only genes that appear in both sources. Denoting by $G_"tumor"$ and $G_"normal"$ the gene sets in the two tables, the final feature set is the intersection $G^* = G_"tumor" ∩ G_"normal"$. Genes with missing names are discarded and duplicate gene columns are removed.

The final design matrix is $X in RR^(n times p), quad p = |G^*|$
and labels are defined as
$ y_i = 0 ("normal"), quad y_i = 1 ("tumor"), $
resulting in $bold(y) in {0,1}^n$.

== Splitting Protocols (Hold-out and Stratified $k$-fold)
The dataset container exposes two splitting schemes used across experiments:

- a reproducible hold-out split with configurable test fraction (default: 80/20) and seed-controlled shuffling;
- stratified $k$-fold cross-validation (default: $k=5$), where stratification approximately preserves the class proportions in each fold.

These protocols are essential in the high-dimensional regime (many more features than samples): performance estimates can vary substantially with the particular split, and stratification reduces the risk of misleading results when classes are imbalanced.

= Experimental Results

This section summarizes the empirical behavior of GLNs on TCGA-BRCA tumor vs. normal classification, with two complementary goals: (i) assess predictive performance in a high-dimensional setting and (ii) characterize how sensitive the method is to optimization choices, which is a prerequisite for any subsequent interpretability analysis.

In particular, the two training regimes introduced in @training-regimes are contrasted: paper-faithful local online OGD versus end-to-end batch optimization (Adam/backpropagation). For both regimes, representative learning dynamics are reported and performance is compared against standard baselines.

== Hyperparameter Configurations
The experiments in this section are run under a small set of controlled hyperparameter choices, with the goal of making training-regime comparisons interpretable rather than exhaustively tuned. The common configuration used for the cross-validation comparison and for the representative training-curve runs is summarized in @tcga_hparams.

Two choices are deliberate. First, several runs use a single epoch (one pass through the data) to directly probe the paper's online-learning claim: meaningful performance should emerge quickly, without relying on repeated passes through a limited cohort. Second, batch size is set to 1 in the most direct comparisons because it maximizes the number of parameter updates per observed sample and makes the distinction between local OGD and backpropagation most visible in the loss/accuracy trajectories. Larger batches (e.g., 10) are additionally included to show how conventional minibatching smooths optimization in the batch regime, even though it departs from the strictly online setting.

== Training Dynamics: Local OGD vs. Backpropagation
On TCGA-BRCA, the separability of the task leads most regimes to achieve high test accuracy, but the training dynamics differ substantially across update rules and learning-rate schedules.

In the local online OGD setting (@tcga_training_curves, subplots #subplot_ref("a") and #subplot_ref("b")), training loss is highly intermittent, with frequent sharp spikes throughout training. This behavior is consistent with the online nature of the update and with the discretization induced by contextual gating: each minibatch (here, a single sample) triggers a potentially large step on the weights of the currently active experts, and small changes in the active-path mixture can translate into abrupt changes in loss.

Learning-rate control is particularly consequential under local updates. The decay schedule (subplot #subplot_ref("a")) yields higher and more stable final accuracy than the constant step-size variant (subplot #subplot_ref("b")), suggesting that step-size decay is important for mitigating the variance introduced by sample-wise updates and context-dependent expert selection.

In contrast, batch backpropagation with batch size 1 (subplots #subplot_ref("c") and #subplot_ref("d")) exhibits a higher loss at the beginning of training but converges more smoothly. The same qualitative pattern holds for the learning-rate schedule: decay (subplot #subplot_ref("c")) outperforms a constant step size (subplot #subplot_ref("d")), indicating that even when optimization is end-to-end, the high-dimensional setting benefits from reducing the effective step size over time.

Finally, batch backpropagation with batch size 10 (subplots #subplot_ref("e") and #subplot_ref("f")) produces the smoothest curves and the fastest convergence in terms of minibatches, at the cost of giving up the strictly online update principle. This contrast motivates the two-regime implementation: local online OGD is retained to test the theoretical learning rule under realistic data, while the batch regime enables controlled ablations and comparisons against standard optimization practice.

#figure(
  pad(x: -0.5cm, grid(
    columns: (50%, 50%),
    gutter: 0pt,
    subplot("a", "assets/training_curves_local_decay.png"), subplot("b", "assets/training_curves_local_constant.png"),
    subplot("c", "assets/training_curves_backprop_decay.png"),
    subplot("d", "assets/training_curves_backprop_constant.png"),

    subplot("e", "assets/training_curves_backprop_10_decay.png"),
    subplot("f", "assets/training_curves_backprop_10_constant.png"),
  )),
  caption: [Training curves on TCGA-BRCA. Each subplot shows loss (top) and test accuracy (bottom) as a function of minibatches. The two columns correspond to learning-rate schedules (decay #subplot_ref("a") #subplot_ref("c") #subplot_ref("e") vs. constant #subplot_ref("b") #subplot_ref("d") #subplot_ref("f")), while the three rows correspond to training regimes: (i) local online OGD, single pass (#subplot_ref("a"), #subplot_ref("b")); (ii) batch backpropagation with batch size 1 (online backprop), single pass (#subplot_ref("c"), #subplot_ref("d")); (iii) batch backpropagation with batch size 10 (standard minibatching), trained for 10 epochs (#subplot_ref("e"), #subplot_ref("f")).
  ],
) <tcga_training_curves>

== Baseline Comparisons
To contextualize GLN performance, the experimental pipeline includes two baselines trained on the same splits used for the GLN runs:
+ Logistic regression as a strong convex model in high-dimensional settings
+ A MLP as a standard nonlinear reference.

The logistic regression baseline is implemented with `sklearn` and uses feature-wise standardization (via `StandardScaler`) followed by `LogisticRegression` with the `lbfgs` solver and an $L_2$ penalty (default inverse regularization strength $C=1$). The MLP baseline matches the GLN hidden-layer sizes for a controlled comparison, uses ReLU nonlinearities and a sigmoid output, and is trained end-to-end with Adam and a linear learning-rate schedule. Inputs are standardized using training-set statistics (as in the GLN pipeline's first standardization step).

Results for these baselines are reported in the cross-validation comparisons in @tcga_model_comparison_cv.

== Cross-Validation Results
Single hold-out splits can overestimate performance when $p$ is large and $n$ is moderate. For this reason, models are also evaluated with stratified $k$-fold cross-validation (default: $k=5$) repeated for $n$ times (default: $n=3$) to assess variability across splits and seeds.

@tcga_model_comparison_cv summarizes these results via accuracy boxplots. Two trends are noteworthy:
+ Baseline methods exhibit relatively tight distributions, reflecting the stability of convex (logistic regression) or well-regularized (MLP) training.
+ GLN variants display larger accuracy dispersion across folds, especially with local OGD, reflecting sensitivity to split-induced shifts and hyperparameter choices. This instability reduces confidence in attribution results: feature attributions are meaningful only when the learned predictors are stable across reasonable resamplings.



#let params = (
  (epochs, 1),
  (batchsize, 1),
  (layeri, [20, 40, 20]),
  (ctxdim, 4),
  (lr, 0.01),
  (lrsched, "const"),
  (seed, 4),
)
#let parameter_figure = [
  #figure(
    parameter_table((params)),
    caption: [Common hyperparameters used for the TCGA-BRCA experiments reported in this section. Parameters not listed here are left at their codebase defaults; the exact CLI options and default values are documented in the repository README.],
  ) <tcga_hparams>
]

#let boxplots = [
  #let subplot = subplot.with(dx: 21%, dy: 85%)
  #figure(
    box(width: 100%, grid(
      columns: (50%, 50%),
      gutter: 0pt,
      subplot("a", "assets/model_comparison_cv_local.png"), subplot("b", "assets/model_comparison_cv_backprop.png"),
    )),
    caption: [Cross-validation comparison across model families. Boxplots report accuracy distributions across repeated stratified $k$-fold splits. The subplot #subplot_ref("a") corresponds to GLNs trained with local online OGD, while #subplot_ref("b") corresponds to GLNs trained with end-to-end backpropagation. In both panels, GLN variants are compared against a logistic regression baseline and a feed-forward MLP baseline (trained with standard backpropagation).],
  ) <tcga_model_comparison_cv>
]

#grid(
  columns: (60%, auto),
  gutter: 10mm,
  inset: 0pt,
  align: horizon,
  boxplots, parameter_figure,
)

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

#show: doc => appendix_template(doc)

== Glossary <glossary>

/ #epochs: number of passes through the training data
/ #batchsize: number of samples per minibatch
/ #layer1, #layer2, #layeri $dots.h$: layer sizes (number of neurons per layer)
/ #ctxdim: context dimension (number of half-space tests per neuron)
/ #lr: learning rate (depends on learning schedule)
/ #lrsched: learning rate schedule (e.g., "decay" for $eta_t = eta_0 / sqrt(t)$, "constant" for $eta_t = eta_0$)
/ #seed: random seed for reproducibility (affects weight initialization and half-space sampling)
