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

// FIXME

== From Backpropagation to Local Learning // FIXME
== The Geometric Mixing Formula // FIXME
== Contextual Gating (Half-Space Partitioning) // FIXME
== Why GLNs Are Special // FIXME
== The GLN Architecture // FIXME

= PyTorch Implementation // FIXME

== Module Structure // FIXME
== Key Implementation Details // FIXME
== Design Choices for Flexibility // FIXME

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
