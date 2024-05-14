# Test Model

`test_model.py` is used to test a model on a dataset. The dataset is expected to be in a specific directory structure and the ground truth data for each image is expected to be in a JSON format. This implementation uses the Unitable model.
`unitable_runner.py` implements all the neccesarry code to run Unitable inferences. It is based on the notebooks provided by the authors [here](https://github.com/poloclub/unitable)

## Setup
To run the unitable model, follow the instructions provided by the authors [here](https://github.com/poloclub/unitable):
> 1. Set up virtual environment (unitable) by running `make .done_venv` in your terminal.
> 2. Download all the model weights from [HuggingFace](https://huggingface.co/poloclub/UniTable/tree/main) by running `make .done_download_weights` in your terminal.
> 3. Try out our demo [Jupyter Notebook](./notebooks/full_pipeline.ipynb) with your own tabular image! Remember to select "unitable" as your notebook kernel. 

## How it works

The script reads each line from the ground_truth file. For every line, an instance of `TestModel` is created with the image path and the ground truth data. The `inference` method of the `TestModel` class is then called with the model to get the prediction. 
The `inference` method runs the code defined in `unitable_runner.py`.

The `compute_metric` method is called to compute the TEDS ( tree-edit-distance-based similarity) for the prediction and the ground truth. The results are stored in a list of dictionaries, where each dictionary contains the directory, image path, prediction, and TEDS.

If there is a JSON decoding error while processing a line, it calls the `print_decoding_error` function with the error.

Finally, it prints the total number of files processed for the directory and saves the results in a CSV file.

