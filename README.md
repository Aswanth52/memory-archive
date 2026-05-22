# Memory Archive

An offline AI-powered photo search and memory system that enables natural language image retrieval, facial recognition, and semantic similarity search using modern computer vision and vector search technologies.

Memory Archive runs completely locally, providing fast performance and privacy-focused image management without relying on cloud services.

---

## Features

* Semantic image search using natural language
* Face recognition and identity grouping
* Fast vector similarity search with FAISS
* Offline-first architecture
* Automatic image embedding generation
* Local image indexing and retrieval
* Recursive folder scanning support
* Automated setup and dependency installation
* Privacy-focused local processing
* Local security validation for file access

---

## Example Queries

```text
busy street
piechart
timetable
aswanth in front of tajmahal
sunset near water
photos of smiling people
```

---

## Tech Stack

### Backend

* Python
* FastAPI

### AI / ML

* OpenCLIP
* FAISS
* DeepFace
* ArcFace
* YuNet Face Detection

### Frontend

* HTML
* CSS
* JavaScript

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

## Requirements

* Windows 10/11
* Python 3.11+
* Internet connection during first setup
* ~3–5 GB free storage for dependencies and AI models
* Recommended: 8 GB RAM or higher

If Python is not installed:

1. Press the Windows key
2. Search for:

```text
cmd
```

3. Open Command Prompt
4. Paste:

```bash
winget install Python.Python.3.11
```

5. Press Enter

---

## Setup

Double click:

```text
setup.bat
```

The setup script automatically:

* installs all required dependencies
* downloads required AI models
* configures the environment
* prepares the application

No manual installation is required.

---

## Run Application

Double click:

```text
run.bat
```

The application will automatically launch locally in your browser.

### Note for First-Time Setup

When running the application for the first time using `run.bat`, startup may take some time because the environment and AI models are still initializing.

In some cases, the browser may open before the local server finishes starting, which can temporarily cause the page to fail loading.

If this happens:

* wait for the terminal process to finish initialization
* give it a few seconds
* refresh/reload the browser tab

The application should then load normally.

---

## Engineering Challenges Solved

### Semantic Search Noise

Initially, semantic search was too loose and occasionally returned unrelated dark images for queries like "sunset".

To improve retrieval quality, cosine similarity threshold filtering and result deduplication logic were implemented.

### Duplicate Image Handling

Some images appeared multiple times because identical files existed across nested backup folders.

Filename tracking and duplicate filtering were added to ensure unique results.

### Recursive Folder Traversal

The initial implementation used `os.listdir()` and missed deeply nested image folders.

This was later upgraded to recursive scanning using `os.walk()`.

### Face Detection Tradeoffs

Multiple face detection approaches were tested:

* OpenCV → lightweight but inaccurate
* RetinaFace → accurate but resource heavy
* YuNet → best balance between speed and accuracy

### Deployment Automation

Instead of packaging massive ML dependencies into an unstable executable, automated setup and runtime scripts were created using:

* `setup.bat`
* `run.bat`
* isolated virtual environments

### Local Security Validation

Canonical path validation was implemented to prevent unintended file access outside allowed directories.

---

## Project Goals

This project was built to explore:

* Multimodal AI systems
* Semantic image retrieval
* Computer vision applications
* Vector databases and embeddings
* Privacy-first local AI applications
* Applied AI engineering workflows

---

## Future Improvements

* Video indexing support
* OCR-based text search
* Conversational AI memory assistant
* GPU acceleration
* Incremental indexing
* Duplicate image detection
* Mobile synchronization
* Docker deployment

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
