# 🖋️ Regeneration of Degraded Handwritten Kannada Letters Using a Conditional Variational Autoencoder (CVAE)

This project proposes a novel solution for restoring degraded handwritten Kannada characters using a **Conditional Variational Autoencoder (CVAE)**. Degraded characters are reconstructed into clean, legible forms by conditioning the model on the target character label, enabling linguistically plausible and accurate restoration. This approach supports better OCR performance and aids in the preservation of cultural documents.

---

## 📌 Key Features

- **CVAE Architecture**: Designed specifically for Kannada script restoration.
- **Conditional Restoration**: Utilizes target character labels for precise regeneration.
- **Script-Aware Reconstruction**: Learns to correct faded strokes or missing regions based on Kannada writing patterns.
- **Synthetic + Real Degradation**: Trained on both synthetic and real-world degraded data.
- **Supports OCR Systems**: Regenerated outputs improve HTR (Handwritten Text Recognition) accuracy.

---

## 🧠 Technologies Used

- PyTorch
- Conditional Variational Autoencoder (CVAE)
- CNN-based Encoder-Decoder
- Reparameterization Trick
- Image Quality Metrics (PSNR, SSIM, MSE)
- OCR Evaluation (Character Accuracy, Word Error Rate)
- GPU acceleration

---

## 🗃️ Dataset

- **Input**: Handwritten Kannada characters (clean and degraded).
- **Degradation Types**: 
  - Ink bleed-through
  - Faded ink
  - Blurred strokes
  - Gaussian noise, scratches, and distortions
- **Format**: Paired (degraded, clean) image samples.
- **Size**: Resized to 128×128, grayscale format.


