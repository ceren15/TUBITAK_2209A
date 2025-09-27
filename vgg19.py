import numpy as np
from keras.api.applications import VGG19
from keras import layers, models
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score

# Verisetinin yolu
base_dir = 'images'

"""
rescale: Piksel değerlerini 0-1 aralığına ölçekler.
width_shift_range: Görüntüleri yatay olarak %20 kaydırabilir.
height_shift_range: Görüntüleri dikey olarak %20 kaydırabilir.
zoom_range: Görüntüleri %20 oranında yakınlaştırabilir.
rotation_range: Görüntüleri 20 derece döndürebilir.
horizontal_flip: Görüntüleri yatay olarak çevirir.
fill_mode: Kenar boşluklarını doldurmak için en yakın piksel değerlerini kullanır.
"""
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

"""
target_size: Görüntüleri 128x128 boyutuna ölçekler.
batch_size: Her seferde 16 görüntü yükler.
class_mode: Çoklu sınıf sınıflandırması için 'categorical' ayarını kullanır.
shuffle: Verileri karıştırır.
seed: Rastgelelik için sabit bir değer belirler.
Test verilerini yükler. Eğitim verisinden farklı olarak, verilerin karıştırılmasını istemez (shuffle=False).
"""
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
"""
Önceden eğitilmiş VGG19 modelini yükler. include_top=False ile son sınıflandırma katmanları hariç tutulur.
"""
# VGG19 modelini yükle
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

"""
GlobalAveragePooling2D: Çıktıyı düzleştirir.
Dense(256): 256 nöronlu bir tam bağlantılı katman ekler.
Dropout(0.3): 0.3 oranında dropout uygular; bu, aşırı öğrenmeyi önler.
Dense: Son sınıflandırma katmanı; sınıf sayısını otomatik olarak alır.
"""
# Modelin üzerine ek katmanlar
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # Dropout oranını artır
predictions = layers.Dense(len(train_generator.class_indices), activation='softmax')(x)  # Sınıf sayısını dinamik olarak al

# Modeli oluştur
model = models.Model(inputs=base_model.input, outputs=predictions)
"""
Temel modelin katmanlarını dondurarak, yalnızca eklenen katmanların eğitilmesini sağlar.
"""
# Katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

# Modeli derleme
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Erken durdurma
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
"""
Modeli eğitim verileriyle eğitir. 20 epoch boyunca, doğrulama verileri ile de test edilir.
"""
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
model.save('vgg19_model.keras')