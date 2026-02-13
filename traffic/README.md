# Traffic Sign Recognition – Experimentation Log

I built a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify 43 different German road signs.

## What I tried

- **Baseline**: a single Conv2D + Pooling + Flatten + Dense output layer.
  → Accuracy ~70%. Underfitting – too shallow.

- **Added a second Conv2D layer** with 64 filters.
  → Accuracy jumped to ~85%. The network could learn more complex patterns.

- **Increased dense layer size** from 64 to 128 units.
  → Slight improvement (87%). Started to see overfitting (training accuracy much higher than validation).

- **Added Dropout (0.5)** after the dense layer.
  → Validation accuracy stabilised around 92–94%. Overfitting reduced.

- **Tried different filter sizes**: 5x5 instead of 3x3.
  → No significant gain; slower training. Stuck with 3x3.

- **Added a third convolutional layer**.
  → Minimal improvement, but training time increased. Kept two layers for efficiency.

- **Adjusted pooling size** from 2x2 to 3x3.
  → Too aggressive; lost spatial information. Reverted to 2x2.

- **Tweaked learning rate** (Adam default works well).
  → No need to change.

## What worked well

- Two convolutional layers (32 and 64 filters) with 3x3 kernels.
- MaxPooling (2x2) after each conv layer.
- A dense layer of 128 units + Dropout 0.5.
- Adam optimizer, categorical crossentropy.

## Final model accuracy

After 10 epochs, test accuracy reaches **~95%**.
This is solid for a simple CNN on this dataset.
