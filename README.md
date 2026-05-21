# Memory Archive

An offline AI-powered photo search and memory system that enables natural language image retrieval, facial recognition, and semantic similarity search using modern computer vision and vector search technologies.

Memory Archive runs completely locally, providing fast performance and privacy-focused image management without relying on cloud services.

---

## Features

- Semantic image search using natural language
- Face recognition and identity grouping
- Fast vector similarity search with FAISS
- Offline-first architecture
- Automatic image embedding generation
- Local image indexing and retrieval
- Modern responsive user interface
- Automated setup and dependency installation
- Privacy-focused local processing

---

## Example Queries

```text
photos at the beach
person wearing black hoodie
images with friends
sunset near water
photos of smiling people
```

---

## Tech Stack

### Backend
- Python
- FastAPI

### AI / ML
- OpenCLIP
- FAISS
- DeepFace
- ArcFace
- YuNet Face Detection

### Frontend
- HTML
- CSS
- JavaScript

---

## How It Works

1. Images are scanned locally
2. OpenCLIP generates semantic embeddings
3. FAISS indexes embeddings for fast retrieval
4. Face detection and recognition create identity profiles
5. User queries are converted into embeddings
6. Similar images are retrieved instantly

---

## Installation

### Option 1 — Clone Repository

```bash
git clone https://github.com/Aswanth52/memory-archive.git
cd memory-archive
```

### Option 2 — Download ZIP

Download the repository ZIP and extract it.

---

## Setup

Double click:

```text
setup.bat
```

The setup script automatically:
- installs all required dependencies
- downloads required AI models
- configures the environment
- prepares the application

No manual installation is required.

---

## Run Application

Double click:

```text
run.bat
```

The application will automatically launch locally in your browser.

---

## Requirements

- Windows 10/11
- Internet connection during first setup

---

## Project Goals

This project was built to explore:

- Multimodal AI systems
- Semantic image retrieval
- Computer vision applications
- Vector databases and embeddings
- Privacy-first local AI applications

---

## Future Improvements

- Video indexing support
- OCR-based text search
- Conversational AI memory assistant
- GPU acceleration
- Incremental indexing
- Duplicate image detection
- Mobile synchronization
- Docker deployment

---

## Privacy

All image processing happens locally on the user's machine.

No images, embeddings, or personal data are uploaded to external servers.

---

## License

MIT License

---

## Author

Aswanth M  
B.Tech Data Science Student  
VIT Chennai

GitHub: https://github.com/Aswanth52