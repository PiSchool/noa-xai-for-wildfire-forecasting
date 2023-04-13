# Explainable Artificial Intelligence (xAI) for Wildfire Forecasting

Due to climate change, we expect an exacerbation of fire in Europe and around the world, with major wildfire events extending to northern latitudes and boreal regions [1]. In this context, it is important to improve our capabilities to anticipate fire danger and understand its driving mechanisms at a global scale. The Earth is an interconnected system, in which large scale processes can have an effect on the global climate and fire seasons. For example, extreme fires in Siberia have been linked to previous-year surface moisture conditions and anomalies in the Arctic Oscillation [2]. In the context of the ESA-funded project SeasFire, we have gathered Earth Observation data related to seasonal fire drivers and created a global analysis-ready datacube for seasonal fire forecasting for the years 2001-2021 at a spatiotemporal resolution of 0.25 deg x 0.25 deg x 8 days [3]. The datacube includes a combination of variables describing the seasonal fire drivers (climate, vegetation, oceanic indices, population density) and the burned areas. Initial studies show the potential of Deep Learning for i) short-term regional [4] and ii) long-term global wildfire forecasting [5]. The goal of this challenge is to develop models that are able to capture global-scale spatiotemporal associations and forecast burned area sizes on a subseasonal to seasonal scale.


## Directory structure
Update appropriately before handing over this repository. You may want to add other directories/files or remove those you don't need.

```
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   ├── raw            <- The original, immutable data dump
│   └── scripts        <- Scripts to download or generate data
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks. Naming convention is a date (for 
│                         ordering) and a short `_` delimited description, 
│                         e.g. `2022-05-18_initial_data_exploration.ipynb`.
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc
│   └── figures        <- Generated graphics and figures
│
├── requirements.txt   <- Required packages (dependencies), e.g. generated 
│                         with `pip freeze > requirements.txt`
│
├── scripts            <- Scripts to train a model, make predictions and so on
│
├── setup.py           <- makes project pip installable (pip install -e .) so 
│                         that `your_package_name` can be imported
└── your_package_name  <- Source code for use in this project
    ├── __init__.py    <- (Optional) Makes `your_package_name` a Python module
    └── *.py           <- Other Python source files (can also be organized in 
                          one or more subdirectories)
```

## How to install
Instructions on how to install the package (can simply amount to setting up a virtual environment and installing the required dependencies).

## Additional data
Instructions on how to download additional data (datasets, pre-trained models, ...) needed for running the code, specifically any kind of data that cannot be stored in this repo due to its large size.

## How to run
Instructions on how to run the code. For example, if the developed code is used as a CLI tool:
```
your_script.py --arg1 val1 --arg2 val2
```
If the code is used as a library/framework, you should provide a quick-start example like the one below:
```python
an_object = MyClass(...)

input_value = "..."
output_value = an_object.do_something(input_value)
```

## The team
This challenge, sponsored by the National Observatory of Athens, was carried out by Giovanni Paolini, and Johanna Strebl as part of the 12th edition of Pi School's School of AI program.
| Giovanni Paolini  | Johanna Strebl | 
| ------------- | ------------- | 
| ![1st team member](https://avatars.githubusercontent.com/u/73278942?v=4) | ![2nd team member](https://media.licdn.com/dms/image/C5603AQEMeBfjHTRBoQ/profile-displayphoto-shrink_800_800/0/1593677172106?e=1686787200&v=beta&t=gwLL1-rT8uTbDGDKomUJlJ8qXGHDddgSun0If5msx9U) | ![3rd team member](https://cdn.icon-icons.com/icons2/2643/PNG/512/male_man_boy_person_avatar_people_white_tone_icon_159357.png)
| Bio for the 1st team member | Bio for the 2nd team member | Bio for the 3rd team member |
| <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [your_name_on_linkedin](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [YokoHono](https://github.com/YokoHono)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [Johanna Strebl](https://www.linkedin.com/in/johanna-strebl/)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) | <img src="https://camo.githubusercontent.com/b079fe922f00c4b86f1b724fbc2e8141c468794ce8adbc9b7456e5e1ad09c622/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6769746875622e737667" width="20"> [your_github_handle](https://github.com/your_github_handle)<br/> <img src="https://camo.githubusercontent.com/c8a9c5b414cd812ad6a97a46c29af67239ddaeae08c41724ff7d945fb4c047e5/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f6c696e6b6564696e2e737667" width="20"> [YokoHono](https://linkedin.com/in/your_linkedin)<br/> <img src="https://camo.githubusercontent.com/35b0b8bfbd8840f35607fb56ad0a139047fd5d6e09ceb060c5c6f0a5abd1044c/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f7376672f747769747465722e737667" width="20"> [@your_twitter_handle](https://twitter.com/your_twitter_handle) |
