import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Rastgelelik kontrolü
tf.keras.utils.set_random_seed(42)

# ==============================================================================
# 3. BÖLÜM: KONVOLÜSYON İŞLEMİNİN KODLANMASI VE FİLTRELERİN GÖRSELLEŞTİRİLMESİ
# ==============================================================================
print("--- 3. BÖLÜM ÇIKTILARI ---")
conv = tf.keras.layers.Conv2D(
    filters=16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal"
)

dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

print("Giriş boyutu:", dummy_input.shape)
print("Çıkış boyutu:", output.shape)
print("Filtre ağırlıklarının boyutu:", weights.shape)
print("Bias boyutu:", biases.shape)
print("İlk filtre ağırlıkları:\n", weights[:, :, 0, 0])

# 16 filtrenin tamamının görselleştirilmesi
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("3. Bölüm: 16 Adet 3x3 Konvolüsyon Filtresinin Başlangıç Ağırlıkları")
for i, ax in enumerate(axes.flat):
    img = ax.imshow(weights[:, :, 0, i], cmap="viridis")
    ax.set_title(f"Filtre {i+1}", fontsize=9)
    ax.axis("off")
plt.colorbar(img, ax=axes.ravel().tolist(), shrink=0.8)
plt.show()

# ==============================================================================
# 4. BÖLÜM: POOLING İŞLEMİNİN İNCELENMESİ
# ==============================================================================
print("\n--- 4. BÖLÜM ÇIKTILARI ---")
input_data = tf.constant(
    [[[1.0, 3.0, 2.0, 4.0], [5.0, 6.0, 1.0, 2.0], [7.0, 2.0, 8.0, 3.0], [4.0, 1.0, 5.0, 9.0]]]
)
input_data = tf.reshape(input_data, (1, 4, 4, 1))

pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
pool_output = pool(input_data)

print("Giriş (4x4):\n", input_data.numpy()[0, :, :, 0])
print("Pooling sonucu (2x2):\n", pool_output.numpy()[0, :, :, 0])

# ==============================================================================
# 5. BÖLÜM: ReLU VE SOFTMAX FONKSİYONLARININ İNCELENMESİ
# ==============================================================================
print("\n--- 5. BÖLÜM ÇIKTILARI ---")
# A. ReLU
x = tf.constant([-5.0, -2.0, -1.0, 0.0, 2.0, 4.0, 6.0])
relu = tf.keras.layers.ReLU()
y = relu(x)
print("ReLU Giriş:", x.numpy())
print("ReLU Çıkış:", y.numpy())

# B. Softmax
logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)
print("Softmax sonuçları:", probabilities.numpy())
print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())

# ==============================================================================
# 6. BÖLÜM: FASHION-MNIST İLE CNN MODELİ OLUŞTURMA VE EĞİTME
# ==============================================================================
print("\n--- 6. BÖLÜM ÇIKTILARI ---")
(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Normalizasyon ve kanal boyutu ekleme
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("Eğitim verisi:", x_train.shape)
print("Test verisi:", x_test.shape)


def build_cnn_model(first_conv_filters=8):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_conv_filters, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Deney 1 (Aynı zamanda 6. Bölümdeki ana model: İlk katmanda 8 filtre)
tf.keras.utils.set_random_seed(42)
model_exp1 = build_cnn_model(first_conv_filters=8)
model_exp1.summary()

start_time_1 = time.time()
history_1 = model_exp1.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1
)
train_time_1 = time.time() - start_time_1

test_loss_1, test_acc_1 = model_exp1.evaluate(x_test, y_test, verbose=0)
print(f"Deney 1 (8 Filtre) -> Test Loss: {test_loss_1:.4f}, Test Accuracy: {test_acc_1:.4f}, Süre: {train_time_1:.2f} sn")

# 6.D Eğitim Sonuçlarının Görselleştirilmesi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history_1.history["accuracy"], label="Training")
plt.plot(history_1.history["val_accuracy"], label="Validation")
plt.title("Accuracy (8 Filtre)")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_1.history["loss"], label="Training")
plt.plot(history_1.history["val_loss"], label="Validation")
plt.title("Loss (8 Filtre)")
plt.xlabel("Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# 8. BÖLÜM: KÜÇÜK BİR DENEY (8 FİLTRE vs 16 FİLTRE KARŞILAŞTIRMASI)
# ==============================================================================
print("\n--- 8. BÖLÜM DENEY SONUÇLARI ---")
tf.keras.utils.set_random_seed(42)
model_exp2 = build_cnn_model(first_conv_filters=16)

start_time_2 = time.time()
history_2 = model_exp2.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1
)
train_time_2 = time.time() - start_time_2

test_loss_2, test_acc_2 = model_exp2.evaluate(x_test, y_test, verbose=0)

print(
    f"\nDeney 1 (8 Filtre)  | Eğitim Acc: {history_1.history['accuracy'][-1]:.4f} | Test Acc: {test_acc_1:.4f} | Süre: {train_time_1:.2f} sn"
)
print(
    f"Deney 2 (16 Filtre) | Eğitim Acc: {history_2.history['accuracy'][-1]:.4f} | Test Acc: {test_acc_2:.4f} | Süre: {train_time_2:.2f} sn"
)