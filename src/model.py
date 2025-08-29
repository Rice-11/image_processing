import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class SignLanguageCNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=24):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_basic_model(self):
        """Create a basic CNN model for sign language recognition"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def create_advanced_model(self):
        """Create an advanced CNN model with residual connections"""
        input_layer = layers.Input(shape=self.input_shape)
        
        # First block
        x = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second block with residual connection
        residual = layers.Conv2D(64, (1, 1), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third block with residual connection
        residual = layers.Conv2D(128, (1, 1), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Dense layers with regularization
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        output_layer = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output_layer)
        self.model = model
        return model
    
    def create_lightweight_model(self):
        """Create a lightweight model for real-time inference"""
        model = models.Sequential([
            # Depthwise separable convolutions for efficiency
            layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.SeparableConv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.SeparableConv2D(96, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Use global average pooling instead of flatten
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with appropriate optimizer and loss function"""
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, model_save_path, patience=10):
        """Get training callbacks for better training"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        return callbacks
    
    def print_model_summary(self):
        """Print detailed model summary"""
        if self.model is None:
            print("No model created yet. Please create a model first.")
            return
            
        print("Model Architecture:")
        print("=" * 50)
        self.model.summary()
        
        # Calculate model size
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
        
        # Estimate model size in MB
        model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
        print(f"Approximate model size: {model_size_mb:.2f} MB")

def create_model(model_type='basic', input_shape=(28, 28, 1), num_classes=24):
    """Factory function to create different types of models"""
    cnn = SignLanguageCNN(input_shape, num_classes)
    
    if model_type == 'basic':
        model = cnn.create_basic_model()
    elif model_type == 'advanced':
        model = cnn.create_advanced_model()
    elif model_type == 'lightweight':
        model = cnn.create_lightweight_model()
    else:
        raise ValueError("Model type must be 'basic', 'advanced', or 'lightweight'")
    
    # Compile the model
    cnn.compile_model(model)
    
    # Print model summary
    cnn.print_model_summary()
    
    return cnn

# Model configuration presets
MODEL_CONFIGS = {
    'basic': {
        'description': 'Basic CNN model with good balance of accuracy and speed',
        'training_time': '30 minutes (CPU) / 5 minutes (GPU)',
        'accuracy': '85-90%',
        'best_for': 'Learning and quick testing'
    },
    'advanced': {
        'description': 'Advanced CNN with residual connections and batch normalization',
        'training_time': '2 hours (CPU) / 20 minutes (GPU)', 
        'accuracy': '90-95%',
        'best_for': 'Production deployment with highest accuracy'
    },
    'lightweight': {
        'description': 'Lightweight model optimized for real-time inference',
        'training_time': '15 minutes (CPU) / 3 minutes (GPU)',
        'accuracy': '80-85%',
        'best_for': 'Real-time applications and edge devices'
    }
}

def print_model_options():
    """Print available model options"""
    print("Available Model Types:")
    print("=" * 50)
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n{model_type.upper()} MODEL:")
        print(f"  Description: {config['description']}")
        print(f"  Training Time: {config['training_time']}")
        print(f"  Expected Accuracy: {config['accuracy']}")
        print(f"  Best For: {config['best_for']}")
    
    print("\nUse create_model('model_type') to create your preferred model.")