---
name: sci-figure-maker
description: Create reproducible, publication-quality scientific figures for AI, machine learning, and recognition research.
version: 0.7.0
---

# Scientific Figure Maker

## Purpose

`sci-figure-maker` is a scientific visualization skill designed for
AI, machine learning, deep-learning, and recognition research.

Its goal is to transform real experimental data and model outputs into
clear, reproducible, publication-quality scientific figures.

## Core Priorities

Always prioritize:

1. Scientific accuracy
2. Reproducibility
3. Appropriate visual representation
4. Clear communication
5. Publication-quality aesthetics

Scientific accuracy must always take priority over appearance.

## Scientific Safety Rules

Never fabricate experimental data.

Never invent model outputs.

Never generate artificial PR, ROC, training, or confidence curves when
the underlying curve data are not available.

Never modify raw experimental measurements merely to improve appearance.

Never hide inconvenient observations.

Never fabricate bounding boxes, segmentation masks, feature maps,
attention maps, CAM results, or other model-derived evidence.

Never treat different uncertainty measures such as SD, SEM, and CI as
interchangeable.

Never claim that a visualization proves causality unless the underlying
experimental design supports that conclusion.

## Primary Research Scope

The default profile focuses on recognition-oriented AI research,
including:

- image classification
- object detection
- semantic segmentation
- instance segmentation
- anomaly detection
- event recognition
- multimodal recognition
- model benchmarking
- ablation studies
- robustness experiments
- deployment and efficiency analysis

The skill should remain sufficiently general to support other
data-driven machine-learning research.

## Main Figure Families

The skill may generate figures for:

### Training

- training loss
- validation loss
- metric curves
- convergence analysis
- learning-rate curves

### Evaluation

- Precision-Recall curves
- ROC curves
- confidence-threshold curves
- confusion matrices
- class-wise metrics

### Benchmarking

- model comparison
- model ranking
- accuracy-efficiency trade-offs
- bubble plots
- Pareto-frontier plots

### Experiment Analysis

- ablation studies
- robustness analysis
- sensitivity analysis
- hyperparameter analysis

### Dataset Analysis

- class distribution
- object-size distribution
- bounding-box statistics
- sample distribution

### Qualitative Analysis

- ground truth vs prediction
- baseline vs proposed method
- detection-result panels
- segmentation-result panels
- failure cases
- feature maps
- attention maps
- CAM / Grad-CAM visualizations

### Paper Figure Organization

- figure planning
- main vs supplementary figure recommendation
- multi-panel figure composition
- panel labels
- publication-size layouts

## Required Workflow

When data are provided:

1. Inspect the input data.
2. Identify the experimental structure.
3. Identify variables and metrics.
4. Determine the scientific question.
5. Select an appropriate figure type.
6. Apply a suitable scientific palette.
7. Generate the figure using reproducible code.
8. Export publication-quality files.
9. Record plotting parameters.
10. Perform figure quality checks.

## Default Output Formats

Whenever applicable, produce:

- SVG
- PDF
- high-resolution PNG
- plotting source code
- processed plotting data
- figure metadata or notes

Vector output should be preferred whenever possible.

## Reproducibility

Every generated scientific figure should be traceable to:

- its source data
- its plotting parameters
- its plotting code
- its figure configuration

The same input and configuration should reproduce the same figure.