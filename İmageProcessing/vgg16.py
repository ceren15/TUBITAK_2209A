import numpy as np
from keras.api.applications import VGG16
from keras import layers, models
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

# Verisetinin yolu
base_dir = 'images'

# Eğitim verisi için veri artırma (augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Test verisi için sadece normalizasyon
test_datagen = ImageDataGenerator(
    rescale=1./255
)

# Eğitim ve test verilerini yükleme
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# VGG16 modelini yükle
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Modelin üzerine ek katmanlar
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # Dropout oranını artır
predictions = layers.Dense(len(train_generator.class_indices), activation='softmax')(x)  # Sınıf sayısını dinamik olarak al

# Modeli oluştur
model = models.Model(inputs=base_model.input, outputs=predictions)

# Katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Erken durdurma
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=20,  # Epoch sayısını artır
    validation_data=test_generator,
    callbacks=[early_stopping]
)

# Test verisi üzerinde doğruluk hesaplama
loss, accuracy = model.evaluate(test_generator)
print(f'Test Doğruluğu: {accuracy*100:.2f}%')

# Test verisi üzerinde tahmin yapma
y_true = test_generator.classes  # Gerçek sınıflar
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# F1 skoru hesaplama
f1 = f1_score(y_true, y_pred_classes, average='weighted')
print(f'F1 Skoru: {f1:.2f}')

# Modeli kaydetme
model.save('vgg16_model.keras')