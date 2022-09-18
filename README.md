# Learning-Based Radiomic Prediction of Type 2 Diabetes Mellitus Using Image-Derived Phenotypes

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-jisoo.chae%40pennmedicine.upenn.edu-blue)](mailto:jisoo.chae@pennmedicine.upenn.edu)

Early diagnosis of Type 2 Diabetes Mellitus is crucial to enable timely therapeutic interventions and lifestyle modifications. As medical imaging data becomes more widely available for many patient populations, we sought to investigate whether image-derived phenotypic data could be leveraged in tabular learning classifier models to predict T2DM incidence without the use of any invasive blood lab measurements. We show that both neural network and decision tree models that use image-derived phenotypes can predict patient T2DM status with recall scores as high as 87.6%. We also propose the novel use of these same architectures as 'SynthA1c encoders,' that are able to output interpretable values mimicking HbA1c blood sugar empirical lab measurements. Finally, we demonstrate that T2DM risk prediction model sensitivity to small perturbations in input feature vector components can be used to predict predictive performance on covariates sampled from previously unseen patient populations.

## Installation

To install and run our code, first clone the `DMT2RiskAssessment` repository.

```
git clone https://github.com/allisonjchae/DMT2RiskAssessment
cd DMT2RiskAssessment
```

Next, create a virtual environment and install the relevant dependencies.

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

At this time, the Penn Medicine BioBank (PMBB) dataset is made available only to researchers affiliated with the University of Pennylsvania at the [PMBB website](https://pmbb.med.upenn.edu). We are in the process of trying to make our dataset available for public use.

## Contact

Questions and comments are welcome. Suggests can be submitted through Github issues. Contact information is linked below.

[Michael Yao](mailto:michael.yao@pennmedicine.upenn.edu)

[Allison Chae](mailto:jisoo.chae@pennmedicine.upenn.edu)

[Hersh Sagreiya](mailto:hersh.sagreiya@pennmedicine.upenn.edu) (*Corresponding Author*)

## Citation

    @misc{yaochae2022dmt2,
      title={Learning-Based Radiomic Prediction of {Type 2 Diabetes Mellitus} Using Image-Derived Phenotypes},
      authors={Yao, Michael S and Chae, Allison and MacLean, Matthew T and Verma, Anurag and Duda, Jeffrey and Gee, James and Torigian, Drew A and Rader, Daniel and Khan, Charles and Witschey, Walter R and Sagreiya, Hersh},
      year={2022},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={},
    }

## License

This repository is MIT licensed (see [LICENSE](LICENSE.md)).