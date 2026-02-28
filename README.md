# iSuara: Edge ML Sign Language Interpreter



## 1. Repository Overview & Team Introduction

Welcome to the official repository for **iSuara**, an Edge-AI Android application built to provide real-time, pocket-sized translation of Bahasa Isyarat Malaysia (BIM) into spoken Malay. This repository contains the complete Android Studio project, including the native Kotlin UI, MediaPipe vision extractors, TensorFlow Lite inference pipeline, and Gemini API integration.

**Team Name:** sudo rm -rf /
**Team Members:** LIM HONG ZHANG, TAN CHEE KEAT, YONG JUN ONN, JITESH A/L MOGANA RAJA

---

## 2. Project Overview

### Problem Statement

In Malaysia, there are **44,000 Deaf and hard-of-hearing individuals**, but only **60 certified interpreters** nationwide. Hiring a human interpreter costs up to RM150/hour and often requires a 3-day wait. As a result, 78% of Deaf individuals never encounter an interpreter when seeking specialized health services, and 70% fear visiting clinics alone due to the risk of being misunderstood. Writing on paper is not a viable substitute because BIM uses a different grammatical structure (Topic-Comment) than spoken Malay, making written text a challenging second language.

### SDG Alignment

iSuara is built to advance the United Nations Sustainable Development Goals:

* **SDG 4 (Quality Education) - Targets 4.5 & 4.a:** Empowers Deaf students to advocate for themselves and participate in inclusive learning environments.
* **SDG 8 (Decent Work & Economic Growth) - Target 8.5:** Breaks workplace communication barriers, allowing seamless idea contribution and productive employment.
* **SDG 10 (Reduced Inequalities) - Target 10.2:** Promotes universal social and economic inclusion by giving the Deaf community an independent, real-time voice.

### Short Description

iSuara is a real-time, native Android application that bridges the communication gap between the Deaf community and the hearing public. It uses on-device Edge Machine Learning to track 98 BIM signs via the smartphone camera and utilizes Google Gemini to grammatically restructure the signs into natural, spoken Bahasa Melayu.

---

## 3. Key Features

* **Real-Time Vision Tracking:** Uses the standard smartphone camera to extract 3D skeletal data without requiring special gloves, depth cameras, or cloud video processing.
* **Speed-Invariant ML Recognition:** A custom AI model that adapts to any signing speed, successfully recognizing highly compressed, fast motion without dropping frames.
* **Semantic AI Translator:** Overcomes the "Topic-Comment" syntax barrier of BIM by inferring hidden context and restructuring disjointed keywords into grammatically perfect Malay.
* **Localized Text-To-Speech:** Instantly vocalizes the translated sentence aloud, specifically targeted for the Malay (`ms_MY`) locale to provide clear, native audio output.

---

## 4. Overview of Technologies Used

### Google Technologies

* **Android Studio & Kotlin Native:** The foundation of our zero-copy architecture, enabling direct access to camera hardware (CameraX) and the device's NPU without the bridge-latency of cross-platform frameworks.
* **Google MediaPipe:** Handles hardware-parallelized skeletal extraction (Pose on GPU, Hands on CPU) to generate 258 dense keypoints per frame.
* **TensorFlow Lite & Android NNAPI:** Runs our custom int8-quantized BiLSTM model locally on the Neural Processing Unit (NPU), taking only ~1-3ms per inference.
* **Gemini 2.5 Flash Lite API:** Acts as our cloud-based semantic brain, transforming raw BIM glosses (e.g., "Market + I + Go") into natural sentences (e.g., "Saya pergi ke pasar") instantly.
* **Google Text-to-Speech (TTS):** The native Android engine used to execute the final audio output.
* **Google Colab:** Our primary environment for model training and evaluating quantitative analytics via Matplotlib.

### Other Supporting Tools

* **Jetpack Compose:** For building a modern, reactive, and overlay-driven UI.
* **Matplotlib:** Used extensively during the Colab training phase to plot cross-entropy loss, accuracy curves, and validate EMA smoothing ratios.

---

## 5. Implementation Details & Innovation

### System Architecture

iSuara utilizes a **Decoupled Edge-Cloud Pipeline**. Heavy visual processing (tracking and sign prediction) happens 100% offline on the Edge, ensuring zero-latency and total user privacy. The Cloud (Gemini API) is triggered only for lightweight semantic translation of text payloads (<1KB).

### Workflow

1. **Capture:** CameraX captures 640x480 video at up to 60 FPS.
2. **Extract:** MediaPipe hardware parallelism tracks body pose (GPU) and dynamically crops hand regions (CPU).
3. **Normalize:** `FrameNormalizer` applies EMA smoothing and extracts 780 physical features (velocity/acceleration) over a 30-frame sliding window.
4. **Predict:** TFLite BiLSTM model evaluates the temporal array to predict one of 98 BIM signs natively on the NPU.
5. **Refine:** Gemini 2.5 Flash Lite restructures the buffered sign tokens into conversational SVO Malay.
6. **Output:** The app displays the text via Compose UI and speaks it aloud via Android TTS.

---

## 6. Challenges Faced

* **Challenge 1: Model Regression & Overheating**
* *Problem:* We initially used a Transformer model. It was too heavy for mobile, causing overheating and dropping framerates to 10 FPS.
* *Solution:* We pivoted to a **Bidirectional LSTM with Dot Attention**. This achieved the same sequence-understanding but is lightweight enough to run in ~3ms on the NPU, bumping our framerate to a smooth 35+ FPS.


* **Challenge 2: High Tracking Latency**
* *Problem:* Real-world conversations were slow because extracting 258 keypoints caused a bottleneck.
* *Solution:* We implemented **Hardware Parallelism**. We split the MediaPipe workload to run the heavy body PoseLandmarker on the GPU while processing the HandLandmarker simultaneously on the CPU.


* **Challenge 3: Restrictive Usable Range**
* *Problem:* Users had to stand rigidly within 50cm of the camera for hands to be recognized.
* *Solution:* We built a **Dynamic Hand-Crop Strategy** combined with shoulder-width normalization. The app uses body wrist coordinates to artificially "zoom in" on the hands, extending our accurate tracking range to 1.5 meters (a 200% increase).



---

## 7. Installation & Setup

### 1. Prerequisites

* Android Studio Ladybug (2024.2) or newer
* JDK 17+
* Android device with API 26+ (Android 8.0)

### 2. Gemini API Key

Create a `local.properties` file in the `android/` directory and add your key:

```properties
GEMINI_API_KEY=your_api_key_here

```

*Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Translation works without this, but will show an error message.*

### 3. Build & Run

```bash
cd android
./gradlew assembleDebug
# Or simply open the project in Android Studio and click Run

```

### 4. First Run Instructions

1. Grant camera permissions when prompted.
2. Point the camera at a person signing BIM.
3. Detected words will appear and build in the bottom buffer.
4. Tap **Translate** to structure the sentence with Gemini AI.
5. Tap **Speak** to hear it via TTS.
6. Tap **Reset** to clear the current buffer.

---

## 8. Future Roadmap

To scale iSuara into a comprehensive accessibility platform, our roadmap is structured into three phases:

* **Short-Term (0–6 Months): Localized Vocabulary Expansion**
Expand the model from 98 to **300+ BIM signs**, targeting essential medical, legal, and emergency vocabulary. We will conduct pilot testing with Malaysian Deaf Associations and integrate an in-app misclassification reporting tool to fine-tune our BiLSTM model.
* **Medium-Term (6–12 Months): B2G Integration & Two-Way Comm**
Transition to a B2B/B2G model by developing an Enterprise Dashboard and deploying iSuara as fixed "Digital Interpreter Kiosks" at hospital and police counters. We will also implement **Two-Way Communication**, utilizing Gemini to convert spoken words from hearing officers back into text or visual BIM avatars.
* **Long-Term (12+ Months): ASEAN Expansion & Technical Scaling**
Retrain our physical feature pipeline to support regional languages like **BISINDO (Indonesia)** and **Thai Sign Language**. Architecturally, we plan to replace the MediaPipe bottleneck with a unified Vision Transformer (ViT) to achieve true device-agnostic hyperscaling on lower-tier Android devices. Finally, we will launch **Semantic Sign Search**, acting as the world's first real-time BIM-to-text dictionary.
