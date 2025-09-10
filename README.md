# AIDL Project: Satellite Image Segmentation for Water Bodies Detection

## 🚀 Project Overview

This project implements various deep learning approaches for **satellite image segmentation** to detect water bodies using the Sentinel-2 satellite dataset. The project explores different **model architectures** and **loss functions** to achieve optimal segmentation performance for water body detection in satellite imagery.

## 📊 Dataset

- **Source**: Sentinel-2 Satellite imagery
- **Total Images**: 2,841 image-mask pairs
- **Format**: JPG images with binary masks (white = water, black = non-water)
- **Resolution**: Resized to 256×256 pixels for processing
- **Dataset Split**: 70% training, 20% validation, 10% testing

## 🗂️ Project Structure

```
AIDL Project/
├── README.md                                                    # This file
├── requirements.txt                                             # Project dependencies
├── finalized_model.sav                                         # Saved trained model
├── scripts/
│   └── preprocess_data.py                                      # Data preprocessing utilities
├── Water Bodies Dataset/
│   ├── Images/                                                 # Input satellite images
│   └── Masks/                                                  # Ground truth masks
└── Notebooks/
    ├── Notebook1_satellite_images_segmentation_final.ipynb            # FCN + Binary Crossentropy
    ├── Notebook1_satellite_images_segmentation_final copy.ipynb       # FCN + Dice Loss
    ├── Notebook1_satellite_images_segmentation_final_Dic_loss.ipynb   # FCN + Dice Loss (v2)
    ├── Notebook1_satellite_images_segmentation_final_focal_loss.ipynb # FCN + Focal Loss
    ├── Notebook2_satellite_image_segmentation_U_net_Diceloss.ipynb    # U-Net + Dice Loss
    ├── Notebook2_satellite_image_segmentation_U_net_IOU_loss.ipynb    # U-Net + IoU Loss
    └── Notebook2_satellite_image_segmentation_tensorflow_Focal_loss.ipynb # U-Net + Focal Loss
```

## 📋 Notebook Analysis & Comparison

### 📖 **Notebook 1 Series: FCN (Fully Convolutional Network) Architecture**

#### 1. **Notebook1_satellite_images_segmentation_final.ipynb**
- **Model Architecture**: FCN (Fully Convolutional Network)
  - Encoder: Conv2D(32) → Pool → Conv2D(64) → Pool
  - Decoder: UpSampling → Conv2D(64) → Conv2D(32) → UpSampling → Conv2DTranspose(1)
- **Loss Function**: Binary Crossentropy
- **Why Binary Crossentropy**: 
  - Standard loss for binary classification problems
  - Pixel-wise classification (water vs. non-water)
  - Simple and computationally efficient
- **Metrics**: Accuracy
- **Performance**: Baseline model - 70.18% validation accuracy
- **Use Case**: Basic segmentation benchmark

#### 2. **Notebook1_satellite_images_segmentation_final copy.ipynb**
- **Model Architecture**: FCN (Same as above)
- **Loss Function**: Dice Loss
- **Why Dice Loss**: 
  - Direct optimization of segmentation overlap
  - Better handling of class imbalance (water vs. non-water pixels)
  - More stable gradients for sparse masks
  - Correlation with evaluation metrics
- **Metrics**: Dice coefficient, Accuracy
- **Improvement**: Better boundary detection and class imbalance handling

#### 3. **Notebook1_satellite_images_segmentation_final_Dic_loss.ipynb**
- **Model Architecture**: FCN
- **Loss Function**: Dice Loss (Alternative implementation)
- **Purpose**: Refined Dice loss implementation with different parameters

#### 4. **Notebook1_satellite_images_segmentation_final_focal_loss.ipynb**
- **Model Architecture**: FCN
- **Loss Function**: Binary Focal Loss (γ=2.0, α=0.25)
- **Why Focal Loss**: 
  - **Hard Example Mining**: Focuses on difficult boundary regions
  - **Class Imbalance Handling**: Superior performance on imbalanced datasets
  - **Reduced Easy Negative Impact**: Down-weights well-classified background pixels
  - **Better Boundary Detection**: Improved precision on water body edges
- **Metrics**: Dice coefficient, Accuracy
- **Expected Benefits**: Better small water body detection and boundary precision

### 📖 **Notebook 2 Series: U-Net Architecture**

#### 5. **Notebook2_satellite_image_segmentation_U_net_Diceloss.ipynb**
- **Model Architecture**: U-Net
  - Encoder: Progressive downsampling (64→128→256→512→1024)
  - Decoder: Upsampling with skip connections
  - Skip Connections: Preserve spatial information
- **Loss Function**: BCE + Dice Loss (Combined)
- **Why U-Net + Dice**: 
  - U-Net's skip connections preserve fine details
  - Dice loss optimizes segmentation overlap
  - Combined approach balances pixel-wise and region-wise optimization
- **Metrics**: Dice coefficient, Accuracy
- **Advantages**: Better feature preservation and detailed segmentation

#### 6. **Notebook2_satellite_image_segmentation_U_net_IOU_loss.ipynb**
- **Model Architecture**: U-Net
- **Loss Function**: IoU (Intersection over Union) Loss
- **Why IoU Loss**: 
  - **Direct Metric Optimization**: Directly optimizes the evaluation metric
  - **Better Boundary Detection**: More effective at preserving object boundaries
  - **Class Imbalance Robustness**: Less sensitive to imbalanced datasets
  - **Geometric Awareness**: Considers spatial overlap between prediction and ground truth
- **Metrics**: IoU coefficient, Dice coefficient, Accuracy
- **Formula**: `IoU Loss = 1 - IoU = 1 - (Intersection / Union)`

#### 7. **Notebook2_satellite_image_segmentation_tensorflow_Focal_loss.ipynb**
- **Model Architecture**: U-Net
- **Loss Function**: Binary Focal Loss (γ=2.0, α=0.25)
- **Why U-Net + Focal Loss**: 
  - Combines U-Net's architectural advantages with Focal loss benefits
  - **Best of Both Worlds**: Detailed feature extraction + hard example mining
  - Optimal for satellite imagery with class imbalance
- **Metrics**: IoU coefficient, Dice coefficient, Accuracy

## 🏆 **Recommendation: Best Model & Configuration**

### **🥇 Winner: Notebook2_satellite_image_segmentation_tensorflow_Focal_loss.ipynb**

**Why This Is The Best Choice:**

#### **Architecture Advantages (U-Net):**
1. **Skip Connections**: Preserve fine spatial details crucial for water boundary detection
2. **Multi-Scale Features**: Captures both local and global context
3. **Proven Performance**: U-Net is the gold standard for medical/satellite image segmentation
4. **Feature Preservation**: Better than FCN at maintaining spatial resolution

#### **Loss Function Advantages (Focal Loss):**
1. **Hard Example Mining**: Automatically focuses on difficult water-land boundaries
2. **Class Imbalance Handling**: Critical for satellite imagery (often more land than water)
3. **Dynamic Weighting**: Adapts training focus based on prediction confidence
4. **Boundary Precision**: Superior performance on thin rivers and small water bodies

#### **Combined Benefits:**
- **Best Segmentation Quality**: U-Net architecture + Focal loss optimization
- **Robust Training**: Handles class imbalance and focuses on challenging regions
- **Practical Performance**: Optimal for real-world satellite imagery applications
- **Scalability**: Can handle various water body sizes and shapes

### **🥈 Runner-up: Notebook2_satellite_image_segmentation_U_net_IOU_loss.ipynb**

**Why Second Choice:**
- U-Net architecture advantages
- Direct IoU optimization
- Good for applications where IoU is the primary evaluation metric
- Slightly less robust than Focal loss for extreme class imbalance

### **🥉 Third Choice: FCN with Focal Loss**

**For Specific Use Cases:**
- When computational resources are limited
- When model simplicity is prioritized
- Still benefits from Focal loss advantages

## 📈 **Performance Comparison**

| Model | Architecture | Loss Function | Key Strengths | Best For |
|-------|-------------|---------------|---------------|----------|
| **U-Net + Focal Loss** ⭐ | U-Net | Focal Loss | Complete solution, hard example mining | **Production use** |
| U-Net + IoU Loss | U-Net | IoU Loss | Direct metric optimization | IoU-focused evaluation |
| U-Net + Dice Loss | U-Net | BCE + Dice | Balanced approach | General segmentation |
| FCN + Focal Loss | FCN | Focal Loss | Lightweight, hard example mining | Resource-constrained environments |
| FCN + Dice Loss | FCN | Dice Loss | Class imbalance handling | Simple baseline |
| FCN + Binary CE | FCN | Binary Crossentropy | Computational efficiency | Quick prototyping |

## 🛠️ **Installation & Setup**

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/shivraj2805/AIDL-Project.git
cd AIDL-Project

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```python
# For best performance, use:
jupyter notebook "Notebook2_satellite_image_segmentation_tensorflow_Focal_loss.ipynb"
```

### Dataset
```python
# For best performance, use:
   Use Drive Link for dataset - "https://drive.google.com/drive/folders/11R8Oj6NT5IlCmrZkjj7NBytGiIkgo0Mh?usp=sharing"
```

## 🔧 **Technical Specifications**

### **System Requirements:**
- **Python**: 3.7+
- **TensorFlow**: 2.8.0+
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional (CUDA-compatible for faster training)
- **Storage**: 5GB+ for dataset and models

### **Model Parameters:**
- **Input Size**: 256×256×3 RGB images
- **Output**: 256×256×1 binary mask
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 8 (adjustable based on GPU memory)

## 📚 **Key Insights & Findings**

### **Loss Function Analysis:**
1. **Focal Loss**: Best for class imbalance and boundary detection
2. **IoU Loss**: Direct metric optimization, good geometric awareness
3. **Dice Loss**: Balanced performance, good for general segmentation
4. **Binary Crossentropy**: Simple but limited for imbalanced data

### **Architecture Analysis:**
1. **U-Net**: Superior feature preservation and detail retention
2. **FCN**: Lightweight but less detailed segmentation
3. **Skip Connections**: Critical for maintaining spatial information

### **Data Insights:**
- **Class Imbalance**: Significant imbalance between water and non-water pixels
- **Boundary Complexity**: Water boundaries are often irregular and challenging
- **Scale Variation**: Water bodies range from large lakes to thin rivers

## 🚀 **Future Improvements**

1. **Advanced Architectures**: DeepLab, PSPNet, or Transformer-based models
2. **Data Augmentation**: Advanced augmentation techniques for better generalization
3. **Multi-Scale Training**: Training on multiple resolutions
4. **Ensemble Methods**: Combining multiple model predictions
5. **Post-Processing**: CRF or morphological operations for cleaner results

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 **Author**

**Shivraj Darekar**
- GitHub: [@shivraj2805](https://github.com/shivraj2805)
- Project: AIDL-Project

---

**💡 Pro Tip**: Start with `Notebook2_satellite_image_segmentation_tensorflow_Focal_loss.ipynb` for the best results in satellite water body segmentation!
