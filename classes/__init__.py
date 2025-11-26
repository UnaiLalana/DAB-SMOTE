"""
classes Package
===============

This package contains the core classes for working with the DAB-SMOTE
(Density Aware Borderline Synthetic Minority Oversampling Technique)
algorithm and its visualization/debugging extension.

Modules
-------
- DAB_SMOTE
    Implements the main DAB-SMOTE algorithm, providing methods for
    generating synthetic samples to address class imbalance problems.
- Visual_DAB
    Extends DAB_SMOTE to provide visualization and debugging capabilities,
    allowing step-by-step inspection of the synthetic data generation process.

Purpose
-------
The package is designed to:
- Provide a data-level resampling approach for imbalanced datasets.
- Enable users to inspect, visualize, and debug the internal steps of
  the DAB-SMOTE algorithm.
- Serve as a foundation for further extensions or modifications of
  DAB-SMOTE.

Notes
-----
- Users can import specific classes directly from this package:

  >>> from classes.DAB_SMOTE import DAB_SMOTE
  >>> from classes.Visual_DAB import Visual_DAB

- The design supports replacing or extending the dataset handling as long
  as the interface matches the expected attributes and methods.
"""
