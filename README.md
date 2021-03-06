# MIO: MicroRNA target analysis system for Immuno-Oncology

In order to identify miRNA target genes of immune-related signatures and pathways in cancer samples, we have developed the user-friendly web platform MIO and the Python toolkit miopy integrating various analyses and visualization methods based on provided or custom bulk miRNA and gene expression data. We include regularized regression and survival analysis and provide information of 40 miRNA target prediction tools as well as a collection of curated immune-related gene signatures and analyses of >30 cancer types based on data from The Cancer Genome Atlas (TCGA). 

- For an online version of the tool, please see [user guide](/man/MIO_Manual.pdf) and [online](https://mio.i-med.ac.at/).

- Please see [setup guide](doc/setupguide.md) for instructions on setting up MIO

MIO is built with three components:

- A website based on Django framework, served as user interface ([web](/web/))
- A set of python scripts include in miopy that generate all the analysis ([repo link](https://github.com/icbi-lab/miopy))


---
_Please cite as:_  

_Monfort-Lanzas P, Gronauer R, Madersbacher L, et al. MIO: MicroRNA target analysis system for Immuno-Oncology. Bioinformatics (Oxford, England). 2022 Jun:btac366. DOI: [10.1093/bioinformatics/btac366](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac366/6596596?login=true). PMID: [35642895](https://pubmed.ncbi.nlm.nih.gov/35642895/)._
---

### Current Analyses

MIO allows the identification of genes affecting the vulnerability of cancer based on synthetic lethal interactions of miRNA target genes. We also integrated several machine learning methods to enable the selection of prognostic and predictive miRNAs and gene interaction network biomarkers. In MIO, users can generate testable hypotheses and identify miRNAs together with their target genes, which may play a potential role as biomarkers or represent direct or susceptible candidates for cancer immunotherapy.

### MIO architecture

![architecture](web/table_architecture.png)
