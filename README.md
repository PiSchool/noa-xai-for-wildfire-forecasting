# Explainable Artificial Intelligence (xAI) for Wildfire Forecasting

## Challenge context
Due to climate change, we expect an exacerbation of fire in Europe and around the world, with major wildfire events extending to northern latitudes and boreal regions [1]. In this context, it is important to improve our capabilities to anticipate fire danger and understand its driving mechanisms at a global scale. The Earth is an interconnected system, in which large scale processes can have an effect on the global climate and fire seasons. For example, extreme fires in Siberia have been linked to previous-year surface moisture conditions and anomalies in the Arctic Oscillation [2]. In the context of the ESA-funded project [SeasFire](https://seasfire.hua.gr/), The team at [NOA](http://orionlab.space.noa.gr/) has gathered Earth Observation data related to seasonal fire drivers and created a global analysis-ready datacube for seasonal fire forecasting for the years 2001-2021 at a spatiotemporal resolution of 0.25 deg x 0.25 deg x 8 days [3]. The datacube includes a combination of variables describing the seasonal fire drivers (climate, vegetation, oceanic indices, population density) and the burned areas. Initial studies show the potential of Deep Learning for i) short-term regional [4] and ii) long-term global wildfire forecasting [5]. The goal of this challenge is to develop models that are able to capture global-scale spatiotemporal associations and forecast burned area sizes on a subseasonal to seasonal scale.

## Problem to solve
Grasp sub-seasonal to seasonal forecasting of global burned area leveraging Explainable AI (XAI) techniques on deep learning models.

## Challenge scope
Which predictors are more important when forecasting at different lead times and ecoregions.
i) Different ecoregions: Is the importance of the variables consistent across different fire regimes? Which variables are important for predicting in different fire regimes (e.g. Mediterranean, Tropics, Savannahs…)
ii) Spatial focus: If the input has a spatial context, in which part of the spatial context does the model pay attention for each variable?
iii) Are the identified explanations physically meaningful/meaningless? Do they reflect physical laws or data artifacts?
iv) Identify some good examples for local explainability. For example, one can search for a major wildfire event where there are known causes, and see if the explainability of the model agrees.

## References
[1] Wu, Chao, et al. "Historical and future global burned area with changing climate and human demography." One Earth 4.4 (2021): 517-530.

[2] Kim, Jin-Soo, et al. "Extensive fires in southeastern Siberian permafrost linked to preceding Arctic Oscillation." Science advances 6.2 (2020): eaax3308.

[3] Alonso, Lazaro, et al. Seasfire Cube: A Global Dataset for Seasonal Fire Modeling in the Earth System. Zenodo, 30 Sept. 2022, p., doi:10.5281/zenodo.7108392.

[4] Kondylatos, Spyros et al. “Wildfire Danger Prediction and Understanding with Deep Learning.” Geophysical Research Letters”, 2022.  doi: 10.1029/2022GL099368

[5] Prapas, Ioannis, et al. "Deep Learning for Global Wildfire Forecasting." arXiv preprint arXiv:2211.00534 (2022).


## Directory structure

```
├── data
│   ├── Biomes_and_GFED <- Data from third party sources
│   ├── images          <- visual results
│   ├── processed       <- average and std used for normalization
│   └── raw             <- Seasfire daatcube tiny examples
│
├── models             <- Trained and serialized models
│
├── notebooks                           <- Jupyter notebooks.
│   ├── binary_segmantation             <- containing GUI for xAI
│   └── fire_size_quantile_regression   <- containing process to train a segmentation model
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- Required packages (dependencies) generated 
│                         with `pip freeze > requirements.txt`
│
├── utils             <- Scripts to train a model, make predictions and visualizations
│
└── setup.py           <- makes project pip installable (pip install -e .) so 
```

## Quickstart GUI

Directly launch the GUI in [Binder](https://mybinder.readthedocs.io) (Interactive jupyter notebook/lab environment in the cloud) or in [Google colab](https://research.google.com/colaboratory/faq.html) (which is much faster for interaction given that it provides GPU resources that makes the xAI models much faster).

[![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tMLrn1UdgDSnZHpH3Gsr_tOinuWcPiOj?usp=sharing)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PiSchool/noa-xai-for-wildfire-forecasting/build_binder?labpath=voila%2Frender%2Fmain%2Fnotebooks%2Fbinary_segmentation%2FGUI_Segmentation_Interpret.ipynb)

## How to install
### Clone this repo

```bash
git clone https://github.com/PiSchool/noa-xai-for-wildfire-forecasting.git
cd noa-xai-for-wildfire-forecasting
```
### Create virtual environment
Once you have cloned and navigated into the repository, you can set up a development environment using `venv` and install all packages from `requirements.txt`.

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

## How to run
Two Jupyter notebook are provided and can be run either in the virtual environment created here, calling `jupyter notebook` and navigating to those files, or opening them into colab.

## The team
This challenge, sponsored by the National Observatory of Athens, was carried out by Giovanni Paolini, and Johanna Strebl as part of the 12th edition of Pi School's School of AI program.
| Giovanni Paolini  | Johanna Strebl | 
| ------------- | ------------- | 
| ![1st team member](https://avatars.githubusercontent.com/u/73278942?v=4) | ![2nd team member](https://media.licdn.com/dms/image/C5603AQEMeBfjHTRBoQ/profile-displayphoto-shrink_800_800/0/1593677172106?e=1686787200&v=beta&t=gwLL1-rT8uTbDGDKomUJlJ8qXGHDddgSun0If5msx9U) | ![3rd team member](https://cdn.icon-icons.com/icons2/2643/PNG/512/male_man_boy_person_avatar_people_white_tone_icon_159357.png)
| Giovanni is an aerospace engineer in his 3rd year as an industrial Ph.D. candidate in the field of remote sensing for agriculture and hydrology. He is currently involved in different projects on the classification of irrigated areas and estimation of water used for agricultural purposes in semi-arid regions, using very high-resolution satellite data and some state-of-the-art ML algorithms. He joined the wildfire challenge at PI school to boost his knowledge of large deep learning models and he is very eager to contribute to the pressing challenge of wildfire prevention. | Johanna is currently in the final year of her Master's degree in Computer Science at the University of Munich, focusing on machine learning and quantum computing. Her main research interest is in using modern technologies to tackle modern problems. After researching hate speech, her most recent focus is now on Earth observation and remote sensing for climate change mitigation, especially forest fire prediction and modeling with AI. At Pi School, Johanna is collaborating with the National Observatory of Athens to research explainable AI for wildfire forecasting. | Bio for the 3rd team member |
| <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [Giov-P](https://github.com/Giov-P)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [Giovanni Paolini](https://www.linkedin.com/in/giovanni-paolini/)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [YokoHono](https://github.com/YokoHono)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [Johanna Strebl](https://www.linkedin.com/in/johanna-strebl/)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [YokoHono](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) |
