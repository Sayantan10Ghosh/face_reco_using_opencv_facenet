# face_reco_using_opencv_facenet
Here's a complete `README.md` for your **Real-Time Face Recognition System using OpenCV and FaceNet**, including camera support, known faces folder setup, dependencies, and an optional reaction on recognition.

---

```markdown
# Real-Time Face Recognition using OpenCV and FaceNet

This project is a real-time face recognition system built with:

- **OpenCV** for webcam access and face detection
- **FaceNet (via facenet-pytorch)** for feature extraction
- **cosine similarity** for face recognition
- **Python**

---

## 🔧 Features

- Face detection using OpenCV's Haar cascade
- Face embeddings using FaceNet (VGGFace2)
- Real-time recognition using webcam
- Displays bounding box and name if face is recognized
- Reaction (console log or optional action) on successful recognition

---

## 📁 Folder Structure

```

project-folder/
├── realtime\_face\_recognition.py
├── known\_faces/
│   ├── messi.jpg
│   ├── messi_1.jpg
│   ├── neymar.jpg
│   ├── neymar_1.jpg
│   ├── ronaldinho.jpg
│   ├── Sayantan.jpg
│   ├── Sayantan Ghosh.jpg
└── README.md

````

- **`known_faces/`**: Contains labeled images of people to recognize.
  - Filename (without extension) is used as the label.
- **`realtime_face_recognition.py`**: Main script
- **`README.md`**: Documentation


📸 Usage

1. Add Known Faces

Put clear frontal face images in the `known_faces/` directory:

```
known_faces/
├── messi.jpg
├─ messi_1.jpg
├── neymar.jpg
├─ neymar-1.jpg
├── ronaldinho.jpg
├─ Sayantan.jpg
├─ Sayantan Ghosh.jpg
```

> The name of each image file (e.g., `Sayantan.jpg`) is used as the label.

### 2. Run the Application

```bash
face_reco_using_opencv_facenet.py
```

* A webcam window will open.
* Detected faces will be labeled with names if matched.
* Press **`q`** to exit.

---

## ✅ Sample Output

* Green rectangle around detected faces
* Name label (e.g., **Alice**) if recognized
* `"Action triggered: Alice recognized!"` message printed on console when matched

---

## 🚀 Custom Reaction on Recognition

Inside the main script, in the recognition loop:

```python
if name != "Unknown":
    print(f"[ACTION] Triggered: {name} recognized!")
    # Example: play sound, send alert, open door, log time, etc.
```

You can customize this to:

* Play a sound (using `playsound`)
* Trigger GPIO (Raspberry Pi)
* Save logs
* Display on GUI
* Send a message or email

---

## 🧠 Future Improvements

* Add GUI (e.g., Streamlit or Tkinter)
* Train on side-profile and rear-left using deep learning or 3D pose estimation
* Save embeddings to avoid recomputing
* Add face registration via webcam


## 🙋‍♂️ Author

* Sayantan Ghosh(https://github.com/Sayantan10Ghosh)

Feel free to contribute or raise issues!
