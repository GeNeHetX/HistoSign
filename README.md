# Prediction of tumor zones and genomic signatures from HES slide

## Initial Data processing

### Data aggregation
First we aggregate the data to several csv files to have a summary of the data in the file `create_cohort.ipynb`. The goal is to have for the three cohort a table resuming the data for each sample. This includes :
- The name of the sample
- The identifier of the patient
- If available, some tabular data about the patient/sample
- The path to the HES slide
- The path to the annotation file, if available
- The signature data

### Data preprocessing
Then we preprocess the slides.

First we extract the tiles from the WSI using `filter_whites_multiscale.py`. It basically segment the slide using Otsu's thresholding method followed by a fex morphological operations. It exports a file containing the coordinates of the tiles as `[z,k,x,y]` where `z` is the zoom level, `k` is the index of the tile, and `x` and `y` are the coordinates of the tile in the slide. Some visualizations are available in the notebook `visualize_tiles.ipynb`.


This step is needed to now performs annotation extraction. For the **MDN** cohort, we extract the different annotations and find which tiles are included in the annotations. We use the script `extract_annotations.py`. For the **Multicentric** cohort, we follow the same step with an additional cleaning of the annotations in the script `extract_annotations_panc_.py`. Some visualizations are available in the notebook `visualize_export_annotations.ipynb`.

Finally we can extract the features for all tiles in all WSI. To do so we use the script `feature_extraction.py`. For each WSI the feature vector is of shape (n_tiles, 3 + embed) where the first columns is the resolution, the 2nd and 3rd are the coordinates of the tile, and the rest are the features.
Remark on installation : the script uses CTransPath whose installation is described on [their GitHub](https://github.com/Xiyue-Wang/TransPath/tree/main)

### Data exploration
In the notebook `count_dataset.ipynb` we explore the data. We count the number of tiles per slide, the number of tiles per patient... We also visualize the distribution of the annotations and also explore the correlation between the different signatures.


## Model training
### Signature prediction
Using the scripts `src/signatures/train.py` and `src/signatures/train_all_sign.py`we train a model for each signature to predict the signature from the tiles. The model used is based on **DeepMIL**. The models are trained using cross validation. Thus we can inspect results and training in the notebook `inspect_training.ipynb` to choose the best model.

### Tumor zone prediction
Using the script `src/tumors/train.py` we train a model to predict the tumor zone from the tiles. The model used is based on a simple **MLP**. The model is trained using cross validation. Thus we can inspect results and training in the notebook `inspect_training_tumors.ipynb` to choose the best model.

## Inference
In order to perform inference, we use the script `src/process_single_WSI/process_wsi.py` to process a single WSI. It will extract the tiles, extract the features, and then use the trained models to predict the signature and the tumor zone. The results are then saved in a csv file and an interactive visualization is available through the script `src/process_single_WSI/display_wsi.py`.