import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.svm import SVC

def adaptive_ml_contours(data, model=None, resolution=100, sigma=1.0, cmap='viridis', normalize_levels=True):
    
    # Validating model parameters
    if model is None:
        model = SVC(kernel='linear')
        # Provide your labeled data and train the model here

    # Smoothening data via Gaussian filter
    smoothed_data = gaussian_filter(data, sigma=sigma)

    # Flatten data and smooth data arr
    X = np.column_stack((data.ravel(), smoothed_data.ravel()))

    # Binary labels for contour vs. non-contour
    # Assume simple thresholding approach based on the median of smoothed data
    threshold = np.median(smoothed_data)
    y = (smoothed_data.ravel() > threshold).astype(int)

    # Training ML model
    model.fit(X, y)

    # Prediction 
    predictions = model.predict(X)

    # Reshaping prediction to match the original data shape
    predictions = predictions.reshape(data.shape)

    # Cal contour lvls based on adaptive contouring
    if normalize_levels:
        total_area = np.sum(smoothed_data)
        contour_levels = np.linspace(0, total_area, resolution)
    else:
        contour_levels = np.linspace(np.min(smoothed_data), np.max(smoothed_data), resolution)

    # Generating contour lines based on predicted labels and adaptive contour lvls
    contour_lines = plt.contour(predictions, levels=contour_levels, cmap=cmap)

    return contour_lines

# Eg.
# Random 2D data
data = np.random.rand(10, 10)

# Plot orig data
plt.imshow(data, cmap='viridis', origin='lower')
plt.colorbar()

# Generating adaptive contouring with ML
contour_lines = adaptive_ml_contours(data, model=None, resolution=100, sigma=1.0, cmap='plasma', normalize_levels=True)

# Show plot
plt.title('Adaptive Contouring with Machine Learning')
plt.show()

