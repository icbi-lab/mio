---
title: "MIO"
output: pdf_document
---

# MIO: MicroRNA target analysis system for Immuno-Oncology

In order to identify miRNA target genes of immune-related signatures and pathways in cancer samples, we have developed the user-friendly web platform MIO and the Python toolkit miopy integrating various analyses and visualization methods based on provided or custom bulk miRNA and gene expression data. We include regularized regression and survival analysis and provide information of 40 miRNA target prediction tools as well as a collection of curated immune-related gene signatures and analyses of >30 cancer types based on data from The Cancer Genome Atlas (TCGA). 

- For an online version of the tool, please see [user guide](https://mio.i-med.ac.at/staticfiles/media/mio_manual.pdf) and [online](https://mio.i-med.ac.at/).

- Please see [setup guide](doc/setupguide.md) for instructions on setting up MIO

MIO is built with three components:

- A website based on Django framework, served as user interface ([repo link](#))
- A set of python scripts include in miopy that generate all the analysis ([repo link](#))


---
_Please cite as:_  


---

### Current Analyses

MIO allows the identification of genes affecting the vulnerability of cancer based on synthetic lethal interactions of miRNA target genes. We also integrated several machine learning methods to enable the selection of prognostic and predictive miRNAs and gene interaction network biomarkers. In MIO, users can generate testable hypotheses and identify miRNAs together with their target genes, which may play a potential role as biomarkers or represent direct or susceptible candidates for cancer immunotherapy.

### MIO architecture



![architecture](doc/architecture.png)

## Current stage of development

RAWG is still in its infancy, we are actively looking for contributors and welcome any comments, suggesttions and ideas. In particular, we are looking for

- Web developer: to furnish our Django website and improve visualisation functionality.
- Server-side developer: our framework for generating workflows need refactor and make integrating new tools easier.
- Researchers with RNASeq experience: what tools do you use which are not currently available on RAWG? How does RAWG compare to your current analysis experience?