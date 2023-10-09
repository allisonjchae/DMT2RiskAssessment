# [SynthA1c: Towards Clinically Interpretable Patient Representations for Diabetes Risk Stratification](https://link.springer.com/chapter/10.1007/978-3-031-46005-0_5)

[![LICENSE](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![CONTACT](https://img.shields.io/badge/contact-michael.yao%40pennmedicine.upenn.edu-blue)](mailto:michael.yao@pennmedicine.upenn.edu)
[![CONTACT](https://img.shields.io/badge/contact-jisoo.chae%40pennmedicine.upenn.edu-blue)](mailto:jisoo.chae@pennmedicine.upenn.edu)

Early diagnosis of Type 2 Diabetes Mellitus (T2DM) is crucial to enable timely therapeutic interventions and lifestyle modifications. As the time available for clinical office visits shortens and medical imaging data become more widely available, patient image data could be used to opportunistically identify patients for additional T2DM diagnostic workup by physicians. We investigated whether image-derived phenotypic data could be leveraged in tabular learning classifier models to predict T2DM risk in an automated fashion to flag high-risk patients without the need for additional blood laboratory measurements. In contrast to traditional binary classifiers, we leverage neural networks and decision tree models to represent patient data as 'SynthA1c' latent variables, which mimic blood hemoglobin A1c empirical lab measurements, that achieve sensitivities as high as 87.6%. To evaluate how SynthA1c models may generalize to other patient populations, we introduce a novel generalizable metric that uses vanilla data augmentation techniques to predict model performance on input out-of-domain covariates. We show that image-derived phenotypes and physical examination data together can accurately predict diabetes risk as a means of opportunistic risk stratification enabled by artificial intelligence and medical imaging.

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

    @inproceedings{yaochae2023syntha1c,
      title={SynthA1c: Towards Clinically Interpretable Patient Representations for Diabetes Risk Stratification},
      authors={Yao, Michael S and Chae, Allison and MacLean, Matthew T and Verma, Anurag and Duda, Jeffrey and Gee, James and Torigian, Drew A and Rader, Daniel and Khan, Charles and Witschey, Walter R and Sagreiya, Hersh},
      year={2023},
      pages={46-57},
      volume={14277},
      doi={10.1007/978-3-031-46005-0_5},
      url={https://link.springer.com/chapter/10.1007/978-3-031-46005-0_5},
      booktitle={Predictive Intelligence in Medicine},
      publisher={Springer},
      venue={Vancouver, Canada},
    }

## License

This repository is MIT licensed (see [LICENSE](LICENSE.md)).
