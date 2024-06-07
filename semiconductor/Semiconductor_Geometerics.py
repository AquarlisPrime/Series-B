import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt

# Load the dataset (dummy example data)
np.random.seed(42)
data_size = 1000
X_data = np.random.rand(data_size, 1) * 10
y_location = np.random.rand(data_size) * 5
y_width = np.random.rand(data_size) * 2

data = pd.DataFrame({'CondensedBinary2DGeometry': X_data.flatten(),
                     'BandGapLocation': y_location,
                     'BandGapWidth': y_width})

# Split the data into features and target variables
X = data['CondensedBinary2DGeometry'].values.reshape(-1, 1)
y_location = data['BandGapLocation'].values.reshape(-1)
y_width = data['BandGapWidth'].values.reshape(-1)

# Normalize the features and target variables
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(pd.concat([pd.Series(y_location), pd.Series(y_width)], axis=1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Define VAE architecture
original_dim = X_train.shape[1]
intermediate_dim = 512  # Increased intermediate dimension for complexity
latent_dim = 20  # Adjusted latent dimension

# Encoder
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
h = BatchNormalization()(h)
h = Dropout(0.4)(h)  # Increased dropout rate for regularization
h = Dense(intermediate_dim//2, activation='relu')(h)
h = BatchNormalization()(h)
h = Dropout(0.4)(h)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder
decoder_h1 = Dense(intermediate_dim//2, activation='relu')
decoder_h2 = Dense(intermediate_dim, activation='relu')
decoder_h_output = decoder_h1(z)
decoder_h_output = BatchNormalization()(decoder_h_output)
decoder_h_output = decoder_h2(decoder_h_output)
decoder_mean = Dense(original_dim, activation='sigmoid')
x_decoded_mean = decoder_mean(decoder_h_output)

# Custom VAE loss layer
class VAELossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var):
        reconstruction_loss = mse(x, x_decoded_mean)
        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return K.mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss)
        return x

loss_layer = VAELossLayer()([inputs, x_decoded_mean, z_mean, z_log_var])
vae = Model(inputs, loss_layer)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Early stopping and learning rate reduction to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Cyclical learning rate
def cyclical_learning_rate(epoch, step_size=10, base_lr=0.0001, max_lr=0.001):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
    return lr

lr_scheduler = LearningRateScheduler(cyclical_learning_rate)

# Train the model
history = vae.fit(X_train, X_train, 
                  epochs=50, 
                  batch_size=64, 
                  validation_split=0.2, 
                  callbacks=[early_stopping, reduce_lr, lr_scheduler])

# Evaluate on test set
test_loss = vae.evaluate(X_test, X_test)
print("Test Loss:", test_loss)

# Additional Metrics
y_pred = vae.predict(X_test)
mse_val = mean_squared_error(X_test, y_pred)
mae_val = mean_absolute_error(X_test, y_pred)
ssim_val = ssim(X_test.flatten(), y_pred.flatten(), data_range=y_pred.max() - y_pred.min(), win_size=3)
psnr_val = psnr(X_test.flatten(), y_pred.flatten(), data_range=y_pred.max() - y_pred.min())
print("MSE:", mse_val)
print("MAE:", mae_val)
print("SSIM:", ssim_val)
print("PSNR:", psnr_val)

# Define the decoder separately for generating new geometries
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h1(decoder_input)
_h_decoded = BatchNormalization()(_h_decoded)
_h_decoded = decoder_h2(_h_decoded)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# Function to generate new geometries given band gap properties
def generate_geometry(band_gap_location, band_gap_width):
    # Scale the band gap properties
    y_input = scaler_y.transform([[band_gap_location, band_gap_width]])
    # Generate the latent vector (assuming a simple linear mapping for illustration)
    z_sample = y_input
    # Generate new geometry
    generated_geometry = vae.predict(z_sample)
    return scaler_X.inverse_transform(generated_geometry)

# Example: Generate geometry for given band gap properties
band_gap_location_example = 0.5
band_gap_width_example = 0.1
new_geometry = generate_geometry(band_gap_location_example, band_gap_width_example)

# Example 2: Inverse Design - Map desired band gap properties to possible geometries
desired_band_gap_location = 0.7
desired_band_gap_width = 0.15

# Generate new geometry based on desired band gap properties
new_geometry_desired = generate_geometry(desired_band_gap_location, desired_band_gap_width)

# Visualize the generated geometry (adjusting for specific data characteristics and research requirements)
plt.figure(figsize=(12, 6))

# Example: Plot as Line Plot (assuming new_geometry is a series of coordinates)
plt.subplot(1, 2, 1)
plt.plot(new_geometry.flatten(), marker='o')
plt.title('Generated Geometry (Line Plot)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Example: Plot as Scatter Plot (assuming new_geometry represents 2D points)
plt.subplot(1, 2, 2)
plt.scatter(new_geometry[:, 0], new_geometry[:, 1], s=50, marker='o')
plt.title('Generated Geometry (Scatter Plot)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)

plt.tight_layout()
plt.show()

# Evaluate test set
test_loss = vae.evaluate(X_test, X_test)
print(f"Test Loss: {test_loss}")
