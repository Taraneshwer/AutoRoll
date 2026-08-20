# AUTOROLL — PRETRAINED WEIGHT AUTHENTICITY AUDIT REPORT

> [!CAUTION]
> **CRITICAL AUDIT FINDING**: All three ML models currently present in `./models/` (`scrfd_10g_bnkps.onnx`, `arcface_iresnet50.onnx`, and `minifasnet_v1.onnx`) are **SYNTHETIC MOCK GRAPHS GENERATED LOCALLY BY AUTOROLL SCRIPTS**. None of the models contain genuine pretrained weights. All three models are classified as **INVALID — NOT PRETRAINED**.

---

## 1. Summary Audit Table

| Model | File Size | Parameters | Source | Genuine Weights | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SCRFD-10G** (`scrfd_10g_bnkps.onnx`) | 2,261 B (0.002 MB) | 500 | Locally Generated (`download_models.py`) | No (Hardcoded Constant Arrays) | **INVALID — NOT PRETRAINED** |
| **ArcFace IResNet50** (`arcface_iresnet50.onnx`) | 77,070,618 B (73.50 MB) | 19,267,584 | Locally Generated (`download_models.py`) | No (Random Gaussian Matrix) | **INVALID — NOT PRETRAINED** |
| **MiniFASNetV1** (`minifasnet_v1.onnx`) | 172 B (0.00016 MB) | 2 | Locally Generated (`download_models.py`) | No (Hardcoded Constant Array) | **INVALID — NOT PRETRAINED** |

---

## 2. Comprehensive File & Structural Inspection

### 2.1. SCRFD Face Detector (`scrfd_10g_bnkps.onnx`)
- **Exact File Size**: `2,261 bytes` (`0.002156 MB` / `2.21 KB`)
- **SHA256 Checksum**: `50c6a85cda80b15954a56497865ca8c4ea686f750dbdbb85f770e24fce1ae079`
- **ONNX IR Version**: `8`
- **ONNX Opset Version**: `ai.onnx` v17
- **Graph Nodes**: `2` nodes (`Identity`, `Identity`)
- **Initializers Count**: `2` initializers (`const_bboxes`, `const_scores`)
- **Total Parameters**: `500` float32 parameters
- **Total Initializer Bytes**: `2,000 bytes` (`0.001907 MB`)
- **Input Tensors**:
  - `input`: shape `[1, 3, 640, 640]`, dtype `FLOAT` (`elem_type=1`)
- **Output Tensors**:
  - `bboxes`: shape `[1, 100, 4]`, dtype `FLOAT` (`elem_type=1`)
  - `scores`: shape `[1, 100, 1]`, dtype `FLOAT` (`elem_type=1`)
- **Graph Operators Used**: `['Identity']`

### 2.2. ArcFace Recognition (`arcface_iresnet50.onnx`)
- **Exact File Size**: `77,070,618 bytes` (`73.500269 MB`)
- **SHA256 Checksum**: `ffc2fbe2e1de34e850a75045ee410f9420d3b9eb6ad8597d9b595bfb0493cd81`
- **ONNX IR Version**: `8`
- **ONNX Opset Version**: `ai.onnx` v17
- **Graph Nodes**: `3` nodes (`Flatten`, `MatMul`, `LpNormalization`)
- **Initializers Count**: `1` initializer (`W`)
- **Total Parameters**: `19,267,584` float32 parameters
- **Total Initializer Bytes**: `77,070,336 bytes` (`73.500000 MB`)
- **Input Tensors**:
  - `input`: shape `[1, 3, 112, 112]`, dtype `FLOAT` (`elem_type=1`)
- **Output Tensors**:
  - `embedding`: shape `[1, 512]`, dtype `FLOAT` (`elem_type=1`)
- **Graph Operators Used**: `['Flatten', 'MatMul', 'LpNormalization']`

### 2.3. MiniFASNet Passive Liveness (`minifasnet_v1.onnx`)
- **Exact File Size**: `172 bytes` (`0.000164 MB`)
- **SHA256 Checksum**: `6d0e1e9e2f945e33b6ec3f7dfc32736f270d302de70ceef353e63f341958cdce`
- **ONNX IR Version**: `8`
- **ONNX Opset Version**: `ai.onnx` v17
- **Graph Nodes**: `1` node (`Identity`)
- **Initializers Count**: `1` initializer (`const_logits`)
- **Total Parameters**: `2` float32 parameters
- **Total Initializer Bytes**: `8 bytes` (`0.000008 MB`)
- **Input Tensors**:
  - `input`: shape `[1, 3, 80, 80]`, dtype `FLOAT` (`elem_type=1`)
- **Output Tensors**:
  - `logits`: shape `[1, 2]`, dtype `FLOAT` (`elem_type=1`)
- **Graph Operators Used**: `['Identity']`

---

## 3. Parameter & Architecture Validation

### 3.1. SCRFD Parameter Validation
- **Official Model Spec**: Genuine InsightFace SCRFD-10G (`scrfd_10g_bnkps.onnx`) contains a deep convolutional backbone (ResNet/MobileNet), Feature Pyramid Network (FPN), multi-stride anchor detection heads (stride 8, 16, 32), and 5 facial landmark keypoint regression heads.
- **Expected Parameters**: ~`3.86M` parameters (~`15.4 MB` FP32 ONNX binary).
- **Actual File Contents**: Contains only `500` parameters. Has zero convolution or BatchNorm layers. Lacks the required `kps` (5 facial landmarks) output node entirely.

### 3.2. ArcFace Parameter Validation
- **Official Model Spec**: Genuine InsightFace ArcFace IResNet50 contains 50 residual bottleneck blocks (Convolution, BatchNorm, PReLU activation), global pooling, and a dense output projection layer.
- **Expected Parameters**: ~`43.5M` parameters (~`166 MB` FP32 ONNX binary).
- **Actual File Contents**: Contains `19,267,584` parameters in a single dense initializer `W` (`37632 x 512`). Completely lacks all 50 IResNet residual convolution blocks and feature extraction stages.

### 3.3. MiniFASNet Parameter Validation
- **Official Model Spec**: Genuine MiniVision Silent-Face MiniFASNetV1/V1SE contains MobileNetV2 inverted residual blocks, Depthwise Separable Convolutions, and Squeeze-and-Excitation (SE) modules.
- **Expected Parameters**: ~`0.41M` to `1.5M` parameters (~`1.6 MB` to `5.8 MB` ONNX binary).
- **Actual File Contents**: Contains `2` parameters total in a 1x2 array. Has zero convolution, pooling, or activation layers.

---

## 4. Weight Value Analysis

| Model | Number of Tensors | Min Value | Max Value | Mean | Std Dev | % Zero Values | % NaN/Inf Values | Weight Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SCRFD-10G** | 2 | `0.95` | `200.0` | `120.190` | `74.529` | `0.00%` | `0.00%` | **C. Constant Weights** |
| **ArcFace IResNet50** | 1 | `-0.055530` | `+0.052275` | `9.0065e-08` | `0.009997` | `0.00%` | `0.00%` | **B. Randomly Initialized** |
| **MiniFASNetV1** | 1 | `0.05` | `0.95` | `0.500` | `0.450` | `0.00%` | `0.00%` | **C. Constant Weights** |

### Initializer Tensor Details:
- **SCRFD-10G**: `const_bboxes` (shape `1x100x4`) is tiled with static box `[100, 100, 200, 200]`. `const_scores` (shape `1x100x1`) contains static `0.95`.
- **ArcFace**: `W` (shape `37632x512`) was generated via `np.random.randn(37632, 512) * 0.01` (Gaussian random noise with mean `0.0` and std `0.01`).
- **MiniFASNetV1**: `const_logits` (shape `1x2`) contains hardcoded floats `[0.05, 0.95]`.

---

## 5. Source & Upstream Verification

### 5.1. SCRFD Face Detector
- **Upstream Repository**: [`deepinsight/insightface`](https://github.com/deepinsight/insightface)
- **Upstream Model Name**: `scrfd_10g_bnkps.onnx` (InsightFace Model Zoo: `detection/scrfd`)
- **Upstream Release/Version**: InsightFace SCRFD v1.0
- **Upstream Download Source**: [InsightFace Model Zoo (GitHub / HuggingFace `deepinsight/insightface`)](https://github.com/deepinsight/insightface/tree/main/detection/scrfd)
- **Local Generation vs Upstream**: **Generated locally** by `scripts/download_models.py::create_scrfd_onnx()`.
- **Upstream Pretrained Checkpoints Exist**: YES (Trained on WIDER FACE dataset).
- **License**: Apache 2.0 / Non-commercial research & academic license.

### 5.2. ArcFace Recognition
- **Upstream Repository**: [`deepinsight/insightface`](https://github.com/deepinsight/insightface)
- **Upstream Model Name**: `glint360k_r50.onnx` / `w600k_r50.onnx` / `ms1mv3_arcface_r50.onnx`
- **Upstream Release/Version**: InsightFace Recognition Release v0.7 / `buffalo_l`
- **Upstream Download Source**: [InsightFace Recognition Model Zoo](https://github.com/deepinsight/insightface/tree/main/python-package/insightface/model_zoo)
- **Local Generation vs Upstream**: **Generated locally** by `scripts/download_models.py::create_arcface_onnx()`.
- **Upstream Pretrained Checkpoints Exist**: YES (Trained on Glint360k / MS1MV3 dataset with 360k+ identities).
- **License**: MIT License / Non-commercial academic license.

### 5.3. MiniFASNet Liveness
- **Upstream Repository**: [`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- **Upstream Model Name**: `1.1_80x80_MiniFASNetV1SE.pth` / `2.7_80x80_MiniFASNetV2.pth` (ONNX export: `minifasnet_v1se.onnx`)
- **Upstream Release/Version**: Silent-Face-Anti-Spoofing v1.0
- **Upstream Download Source**: [Silent-Face-Anti-Spoofing Releases](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/master/resources/anti_spoof_models)
- **Local Generation vs Upstream**: **Generated locally** by `scripts/download_models.py::create_minifasnet_onnx()`.
- **Upstream Pretrained Checkpoints Exist**: YES (Trained on CASIA-SURF and SiW datasets).
- **License**: Apache 2.0 License.

---

## 6. Checksum Comparison

| Model | Local File SHA256 Checksum | Upstream Checksum Provided | Match Status |
| :--- | :--- | :--- | :--- |
| **SCRFD-10G** | `50c6a85cda80b15954a56497865ca8c4ea686f750dbdbb85f770e24fce1ae079` | Documented in InsightFace Releases | **MISMATCH (Synthetic local file)** |
| **ArcFace** | `ffc2fbe2e1de34e850a75045ee410f9420d3b9eb6ad8597d9b595bfb0493cd81` | Documented in InsightFace Model Zoo | **MISMATCH (Synthetic local file)** |
| **MiniFASNet** | `6d0e1e9e2f945e33b6ec3f7dfc32736f270d302de70ceef353e63f341958cdce` | Documented in Silent-Face Releases | **MISMATCH (Synthetic local file)** |

---

## 7. Model Functionality Testing

Empirical inference tests were conducted using sample face images (`data/test_samples/sample_face.jpg`, `data/raw_datasets/sample_subset/student_id_001/sample_01.jpg`, `sample_02.jpg`, and `student_id_002/sample_01.jpg`):

### 7.1. SCRFD Face Detection Test
- **Detection Results**: Failed to detect actual faces. Returns hardcoded bounding box `[100, 100, 200, 200]` and confidence `0.95` for every input image.
- **Landmarks Output**: Missing (0 landmark keypoints returned; expected 5 points `(x, y)`).

### 7.2. ArcFace Recognition Discriminative Test
- **Same Person Cosine Similarity** (Student 1 Image 1 vs Student 1 Image 2): `1.000000` (identical file copy test).
- **Different Person Cosine Similarity** (Student 1 Image 1 vs Student 2 Image 1): `0.999587`.
- **Discriminative Evaluation**: **FAILED**. Random projection maps all RGB image inputs to almost identical vector directions in 512-D space (cosine similarity > 0.999 for different identities). The model is completely non-discriminative.

### 7.3. MiniFASNet Liveness Test
- **Real Face Crop Logits**: `[[0.0500, 0.9499]]` (Probabilities: Real=`0.2891`, Spoof=`0.7109`).
- **Random Noise / Spoof Input Logits**: `[[0.0500, 0.9499]]` (Probabilities: Real=`0.2891`, Spoof=`0.7109`).
- **Liveness Evaluation**: **FAILED**. Returns identical static output regardless of image content. Zero anti-spoofing capability.

---

## 8. Replacement Models & Official Download Sources

All three synthetic models must be replaced with official pretrained ONNX weights:

1. **SCRFD Face Detector**:
   - **Target Model**: `scrfd_10g_bnkps.onnx` (~15.4 MB)
   - **Upstream Source**: InsightFace Model Zoo ([GitHub `deepinsight/insightface`](https://github.com/deepinsight/insightface/tree/main/detection/scrfd) / Hugging Face `insightface`)
   - **Required Outputs**: `stride_8`, `stride_16`, `stride_32` score, bbox, and 5-point landmark keypoints tensors (`kps`).

2. **ArcFace Face Recognition**:
   - **Target Model**: `glint360k_r50.onnx` or `w600k_r50.onnx` (~166 MB)
   - **Upstream Source**: InsightFace Model Zoo ([GitHub `deepinsight/insightface`](https://github.com/deepinsight/insightface/tree/main/python-package/insightface/model_zoo))
   - **Required Architecture**: Full 50-layer IResNet backbone with 512-D normalized output embedding.

3. **MiniFASNet Liveness**:
   - **Target Model**: `minifasnet_v1se.onnx` or `minifasnet_v2.onnx` (~1.6 MB to ~5.8 MB)
   - **Upstream Source**: Silent-Face-Anti-Spoofing ([GitHub `minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing))
   - **Required Architecture**: MobileNetV2/MiniFASNet with Squeeze-and-Excitation layers and binary live/spoof logits output.

---

## 9. Recommended Next Steps

1. **Update Download Automation Script**: Replace synthetic ONNX graph generators in `scripts/download_models.py` with an automated HTTP downloader function that fetches genuine pretrained ONNX binaries from official upstream release mirrors (e.g., HuggingFace hub or GitHub release assets).
2. **Re-run Validation & Functional Test Suites**: After downloading genuine pretrained weights, run `scripts/validate_models.py` and `scripts/test_real_ml_pipeline.py` to confirm face detection bounding boxes, landmark alignment, face recognition cosine similarity separation (same person > 0.65, different person < 0.20), and liveness spoof rejection.
