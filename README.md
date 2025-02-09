# Image Classification with TinyVGG

This project uses a TinyVGG neural network to classify images into two classes: "ticker" and "no_ticker".
Read more about the TinyVGG here: https://arxiv.org/pdf/2004.15004

## Getting Started
### Data Setup
To use this project, you'll need to set up a `data` directory in the root of the project. This directory should contain all the images you want to train the model on.

**Name the image files correctly:** Image filenames should start with either "ticker" or "no_ticker" to indicate their class.

### Training the Model
Once your data is set up, you can train the model by running:
```bash
python3 train.py
```
This will train the TinyVGG model on your dataset.

*Running the Application*
After training the model, you can run the application by executing:
```
bash
python3 main.py
```
This will start a simple web application that allows you to upload images and classify them using the trained model.

# Requirements
- Python 3.x
- PyTorch
- torchvision
