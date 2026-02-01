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

// FIXME

== Project Objective // FIXME
== Motivation // FIXME
== Problem Statement // FIXME
== Report Structure // FIXME

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
