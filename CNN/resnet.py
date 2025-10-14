import sys
sys.path.append("../")
from core.model_structure import Deep_learning_Model
from utils.activations import *
from CNN.Vectorised_Cnn_operations import *
from CNN.Vectorised_Cnn_operations.Vec_cnn_Layers import *
from CNN.Loop_based_cnn.Cnn_layers import *

Resnet50 = Deep_learning_Model("adam", "Crossentropy")

Resnet50.add_layers([

    # --- Conv1 initial (224→112 ou 64→32) ---
    vec_Conv2D_layer(3, 64, (7, 7), stride=2, padding=3, initializer="HeNormal"),
    Batch_norm_layer(64),
    LeakyRelu(0.01),

    # --- MaxPool réduit encore (32→16) ---
    Vec_Max_pool_layer((3, 3), stride=2),

    # ---- Bloc résiduel 1 ---- (16x16)
    vec_Conv2D_layer(64, 64, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(64),
    LeakyRelu(0.01),
    vec_Conv2D_layer(64, 64, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(64),
    LeakyRelu(0.01),

    # ---- Bloc résiduel 2 ---- (réduction : stride=2 → 8x8)
    vec_Conv2D_layer(64, 128, (3, 3), stride=2, padding=1, initializer="HeNormal"),
    Batch_norm_layer(128),
    LeakyRelu(0.01),
    vec_Conv2D_layer(128, 128, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(128),
    LeakyRelu(0.01),

    # ---- Bloc résiduel 3 ---- (8x8 conservé)
    vec_Conv2D_layer(128, 256, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(256),
    LeakyRelu(0.01),
    vec_Conv2D_layer(256, 256, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(256),
    LeakyRelu(0.01),

    # ---- Bloc final (8x8 conservé, plus de stride=2 !) ----
    vec_Conv2D_layer(256, 256, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(256),
    LeakyRelu(0.01),
    vec_Conv2D_layer(256, 128, (3, 3), stride=1, padding=1, initializer="HeNormal"),
    Batch_norm_layer(128),
    LeakyRelu(0.01),
])
