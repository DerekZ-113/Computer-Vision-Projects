# Mini-Project 10: Customized U-Net

## Team Members

- **Derek Zhang**
- **Yichi Zhang**
- **Yiling Hu**
- **Ziyang Liang**

---

## Project Overview

This project applies a **customized U-Net model** for road segmentation tasks using a dataset of urban images with ground truth masks. The U-Net model has been tailored to achieve high segmentation accuracy, particularly in identifying road areas from images. We optimized the model with hyperparameter tuning and additional techniques to improve performance and visualization.

## Setup Instructions

Ensure Python 3.x is installed on your system. Install the required libraries by running:

```bash
pip install tensorflow opencv-python-headless numpy matplotlib scikit-learn
```

## Dataset

### Dataset Preparation

The dataset is expected to be a zip file named `city_dataset.zip` stored in the [Google Drive root directory](https://drive.google.com/drive/folders/15-kAGWv6w-dQvaKmbjGABgWOC9qtyD_I).

- **Dataset Location**: `city_dataset.zip` (downloaded and stored locally after unzipping).
- **Paths in Code**:
  - Images: `leftImg8bit_trainvaltest/train`
  - Masks: `gtFine_trainvaltest/train`

### Unzip the Dataset

In the Google Colab environment:

```python
!unzip -oq '/content/drive/MyDrive/city_dataset.zip' -d '/content/'
```

## Model Architecture

You can find the training code in the same [Google Drive root directory](https://drive.google.com/drive/folders/15-kAGWv6w-dQvaKmbjGABgWOC9qtyD_I) named **SecondBatchCode.ipynb**.
The model is a U-Net architecture with:

- **Input Shape**: 512x512x3
- **Batch Size**: 4 (optimized for the input size)
- **Optimizer**: Adam with learning rate of 1e-4
- **Loss Function**: Combination of Binary Cross-Entropy and Dice Loss
- **Metrics**: IoU (Intersection over Union) and Accuracy

The U-Net includes data augmentation parameters:

- Rotation, width, and height shifts
- Zoom, brightness, and horizontal flip adjustments

## Training and Validation

- **Training/Validation Split**: The data is split into 80% for training and 20% for validation.
- **Batch Size**: 4 (to manage memory with 512x512 image inputs).
- **Epochs**: Trained for 50 epochs with early stopping and learning rate reduction on plateau.

### Augmentation Parameters

We used the following data augmentation for training:

- **Rotation Range**: 15 degrees
- **Width/Height Shift Range**: 10%
- **Zoom Range**: 20%
- **Horizontal Flip**: Enabled
- **Brightness Range**: 0.8 to 1.2

## Model Training

To train the model, run the following command:

```python
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks
)
```

### Callbacks

- **ModelCheckpoint**: Saves the best model: unet_best_model.keras.
- **EarlyStopping**: Stops training if no improvement in 10 epochs.
- **ReduceLROnPlateau**: Reduces learning rate by 50% if no improvement in 5 epochs.

## Evaluation

The model's performance was evaluated using IoU and accuracy metrics:

```python
test_loss, test_accuracy, test_iou = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
print(f'Test IoU: {test_iou}')
```

## Visualization of Results

The code provides functions to visualize the original images, ground truth masks, and predicted masks.
[Results of visualization](https://drive.google.com/drive/folders/15-kAGWv6w-dQvaKmbjGABgWOC9qtyD_I)

1. **Display Sample Images and Masks**: Shows random samples of images and masks before training.
2. **Plot Training Metrics**: Plots training and validation loss and IoU across epochs.
3. **Display Predictions**: Generates side-by-side comparisons of images, ground truth masks, and predicted masks with IoU scores.

To display predictions with IoU:

```python
display_predictions(model, X_test, y_test)
```

## Files in Repository: [Google Drive Link](https://drive.google.com/drive/folders/15-kAGWv6w-dQvaKmbjGABgWOC9qtyD_I)

- `city_dataset.zip`: Compressed dataset for training and testing.
- `unet_best_model.keras`: Saved model with the best performance.
- `second_batch_code.rtf`: The Python script containing all steps for loading data, training, and evaluating the model.

## Improvements made

[Google Drive Link](https://drive.google.com/drive/folders/180VYhzA12a1hKtNVoln0vDxOjmY0rgKj)

We were actually training the models for four times (reaching limitaion) expecting to find a better result. However, due to some problems related to input, the results of third and fourth batches got better testing accuracy with totally black ground truth and IoU as 0. Thus, we are handing in the second batch.
