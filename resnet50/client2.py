import flwr as fl
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import shutil
import random


dataset_path = "dataset2"
save_dir = 'client2'
train_ratio = 0.6
test_ratio = 0.2
valid_ratio = 0.2

def split_data(data_dir, save_dir, train_ratio, test_ratio, valid_ratio):
    train_dir = os.path.join(save_dir, 'train')
    valid_dir = os.path.join(save_dir, 'valid')
    test_dir = os.path.join(save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = os.listdir(data_dir)
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)

        train_class_dir = os.path.join(train_dir, class_name)
        valid_class_dir = os.path.join(valid_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(valid_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        files = os.listdir(class_dir)
        random.shuffle(files)

        total_files = len(files)
        train_split = int(train_ratio * total_files)
        test_split = int(test_ratio * total_files)

        train_files = files[:train_split]
        test_files = files[train_split:train_split+test_split]
        valid_files = files[train_split+test_split:]

        for file in train_files:
            shutil.copyfile(os.path.join(class_dir, file), os.path.join(train_class_dir, file))
        for file in test_files:
            shutil.copyfile(os.path.join(class_dir, file), os.path.join(test_class_dir, file))
        for file in valid_files:
            shutil.copyfile(os.path.join(class_dir, file), os.path.join(valid_class_dir, file))

    print('Data split into train, valid, and test directories.')

split_data(dataset_path, save_dir, train_ratio, test_ratio, valid_ratio)


img_size =(224,224)
batch_Size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(save_dir, 'train'),
    target_size=img_size,
    batch_size=batch_Size,
    class_mode='categorical',
    shuffle = True
)
test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(save_dir, 'test'),
    target_size=img_size,
    batch_size=batch_Size,
    class_mode='categorical',
    shuffle = False
)
val_datagen=ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    os.path.join(save_dir, 'valid'),
    target_size=img_size,
    batch_size=batch_Size,
    class_mode='categorical',
    shuffle=False
)

def build_model():
    # Use pre-trained weights for transfer learning
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))   
    
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Replace Flatten with GAP for better generalization
    x = Dense(256, activation='relu')(x)
    x =  Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    # Define the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data, test_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Reduce patience
        history = self.model.fit(
            self.train_data,
            batch_size=64,
            epochs=50,  # Reduce epochs
            validation_data=self.val_data,
            callbacks=[early_stop]
        )

        # Save training history
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
        })
        results_dir = "client2"
        os.makedirs(results_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)

        # Save training curves
        plt.figure(figsize=(10,5))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Loss per Epoch')
        plt.savefig(os.path.join(results_dir, "loss_curve.png"))

        plt.figure(figsize=(10,5))
        plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], label='Train Accuracy')
        plt.plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title('Accuracy per Epoch')
        plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))

        # Save model
        self.model.save(os.path.join(results_dir, "final_model.h5"))

        return self.model.get_weights(), self.train_data.samples, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data)

        # Predictions and metrics
        pred_probs = self.model.predict(self.test_data)
        y_pred = np.argmax(pred_probs, axis=1)
        y_true = self.test_data.classes

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
    


        # Save metrics
        with open(os.path.join("client2", "final_metrics.txt"), "w") as f:
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall: {rec:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            # f.write(f"\nAUC: {roc_auc:.4f}\n")

        return loss, self.test_data.samples, {"accuracy": accuracy}


model = build_model()
client = FlowerClient(model, train_generator, val_generator, test_generator)
fl.client.start_client(
    server_address="localhost:8080",
    client=client
)