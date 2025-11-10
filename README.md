# 🚗 Smart Parking Spot Detection using CNN and OpenCV

## 📘 Project Overview

This project detects empty and occupied parking spots from an image using a **Convolutional Neural Network (CNN)** trained on parking lot images. It combines **OpenCV** for image handling and visualization with **Keras/TensorFlow** for model training and prediction.

Users can draw parking spot regions manually on an image, train a model using sample parking lot images, and then run inference to detect empty spaces in a given parking lot image.

---

## 🧰 Tools & Frameworks Used

* **Python 3.10+**
* **OpenCV** – Image processing and region selection
* **TensorFlow / Keras** – CNN model creation and training
* **NumPy** – Numerical computation
* **Scikit-learn** – Data splitting for training/testing

---

## 🧠 Model Architecture

A CNN-based binary classifier trained to recognize **empty** vs **occupied** parking spaces.

**Layers used:**

* Conv2D → ReLU activation
* MaxPooling2D → Dimensionality reduction
* Dropout → Prevent overfitting
* Dense layers → Classification

**Output:** 2-class softmax (empty, occupied)

---

## 🖼️ Workflow

1. **Mark Parking Spots** using OpenCV mouse drawing:

   ```python
   cv2.setMouseCallback('image', draw_rectangle)
   ```

   This prints top-left and bottom-right coordinates of each parking region.

2. **Train Model** using custom dataset directories:

   ```bash
   /matchbox_cars_parkinglot/empty
   /matchbox_cars_parkinglot/occupied
   ```

3. **Save Model**:

   ```python
   model.save('emptyparkingspotdetectionmodel.h5')
   ```

4. **Run Detection** on any image by loading the model and looping through pre-defined coordinates.




---

## 🔍 Challenges Faced

### 1. **Data Collection**

* Finding labeled datasets for parking spots was difficult.
* Created a small dataset manually using toy car images.

### 2. **Generalization Problem**

* The model performed well on similar images but struggled with new ones (different lighting, angles, etc.).
* Future improvement: augment data and fine-tune model for diverse parking environments.

### 3. **Training Time**

* 20 epochs took several minutes depending on dataset size.
* GPU acceleration helped reduce training time.

### 4. **Image Scaling Issue**

* Uploaded test images appeared very large when displayed.
* Fixed by resizing before display:

  ```python
  current_image = cv2.resize(current_image, (960, 540))
  ```

---
### Dataset Link - "https://www.kaggle.com/datasets/ljkeller/matchbox-car-parking-occupancy/code"

## 🧪 Example Output

Green rectangles → Empty spots
Red rectangles → Occupied spots
Displayed on top of the original parking lot image.

---

## 📈 Future Improvements

* Add **real-time video stream support**.
* Implement **web dashboard** for live parking availability monitoring.


