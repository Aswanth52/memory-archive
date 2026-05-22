import sys
import os
import time

if getattr(sys, 'frozen', False):
    APP_ROOT = os.path.dirname(sys.executable)
else:
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

os.environ["HF_HOME"] = os.path.join(APP_ROOT, "ai_models_cache")

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import open_clip
import torch
import faiss
import numpy as np
import pickle
from PIL import Image
import json
from deepface import DeepFace
import base64
import shutil
import re
import threading

app = FastAPI()
index_lock = threading.Lock()

print("Loading search engine...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
print("Ready!")

INDEX_FILE      = os.path.join(APP_ROOT, "photos.index")
PATHS_FILE      = os.path.join(APP_ROOT, "photo_paths.pkl")
SETTINGS_FILE   = os.path.join(APP_ROOT, "settings.json")
FACES_DIR       = os.path.join(APP_ROOT, "registered_faces")
FACES_FILE      = os.path.join(APP_ROOT, "faces.pkl")
FACE_INDEX_FILE = os.path.join(APP_ROOT, "face_index.pkl")

INDEX_FILE_TMP      = os.path.join(APP_ROOT, "photos.index.tmp")
PATHS_FILE_TMP      = os.path.join(APP_ROOT, "photo_paths.pkl.tmp")
FACE_INDEX_FILE_TMP = os.path.join(APP_ROOT, "face_index.pkl.tmp")

os.makedirs(FACES_DIR, exist_ok=True)

def is_path_safe(target_path: str, allowed_folders: list) -> bool:
    """Prevent path traversal: ensures path is within allowed folders or FACES_DIR."""
    try:
        abs_target = os.path.abspath(target_path)
        all_allowed = list(allowed_folders) + [os.path.abspath(FACES_DIR)]
        for folder in all_allowed:
            if abs_target.startswith(os.path.abspath(folder)):
                return True
    except Exception:
        pass
    return False

_faces_cache      = {}
_face_index_cache = {}

def load_faces():
    if os.path.exists(FACES_FILE):
        with open(FACES_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_faces(faces):
    global _faces_cache
    with open(FACES_FILE, "wb") as f:
        pickle.dump(faces, f)
    _faces_cache = faces

def load_face_index():
    if os.path.exists(FACE_INDEX_FILE):
        with open(FACE_INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_face_index(face_index):
    global _face_index_cache
    with open(FACE_INDEX_FILE, "wb") as f:
        pickle.dump(face_index, f)
    _face_index_cache = face_index

def face_similarity(query_embeddings, photo_emb):
    photo_emb = np.array(photo_emb, dtype=np.float32)
    photo_norm = np.linalg.norm(photo_emb)
    if photo_norm < 1e-6:
        return 0.0
    photo_emb = photo_emb / photo_norm
    best = 0.0
    for qe in query_embeddings:
        qe = np.array(qe, dtype=np.float32)
        qe_norm = np.linalg.norm(qe)
        if qe_norm < 1e-6:
            continue
        qe = qe / qe_norm
        sim = float(np.dot(qe, photo_emb))
        if sim > best:
            best = sim
    return best

def match_name_in_query(query: str, faces: dict):
    sorted_names = sorted(faces.keys(), key=len, reverse=True)
    for name in sorted_names:
        pattern = r'(?<![a-zA-Z])' + re.escape(name) + r'(?![a-zA-Z])'
        if re.search(pattern, query, re.IGNORECASE):
            clean = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
            return name, clean
    return None, query.strip()

index        = None
photo_paths  = []
indexing_status = {
    "running": False, "progress": 0, "total": 0,
    "done": False, "folder": "", "phase": "",
    "skipped": 0, "new": 0
}

def load_index():
    global index, photo_paths
    if os.path.exists(INDEX_FILE) and os.path.exists(PATHS_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(PATHS_FILE, "rb") as f:
            photo_paths = pickle.load(f)
        print(f"Loaded {len(photo_paths)} photos.")

def save_settings(folders: list):
    with open(SETTINGS_FILE, "w") as f:
        json.dump({"folders": folders}, f)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
            if "folder" in data:
                return [data["folder"]]
            return data.get("folders", [])
    return []

def run_indexing(folders: list):
    global index, photo_paths, indexing_status, _face_index_cache
    indexing_status.update({
        "running": True, "done": False, "folder": ", ".join(folders),
        "skipped": 0, "new": 0
    })

    existing_paths = set(photo_paths)
    existing_faces = dict(_face_index_cache)

    images_to_process = []
    seen_paths = set()

    for folder in folders:
        if not os.path.exists(folder):
            continue
        try:
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        full_path = os.path.join(root, f)
                        if full_path not in seen_paths:
                            seen_paths.add(full_path)
                            images_to_process.append((full_path, f))
        except:
            pass

    indexing_status["total"] = len(images_to_process)

    new_images = []
    skipped    = 0
    for path, filename in images_to_process:
        if path in existing_paths:
            skipped += 1
        else:
            new_images.append((path, filename))

    indexing_status["skipped"] = skipped
    indexing_status["new"]     = len(new_images)

    if not new_images:
        indexing_status.update({"running": False, "done": True, "phase": ""})
        return

    # Phase 1: CLIP Description Processing
    indexing_status["phase"]    = "clip"
    indexing_status["progress"] = 0
    new_embeddings = []
    new_paths      = []

    for i, (path, filename) in enumerate(new_images):
        try:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                features = model.encode_image(img)
                features /= features.norm(dim=-1, keepdim=True)
            new_embeddings.append(features.cpu().numpy())
            new_paths.append(path)
        except:
            pass
        indexing_status["progress"] = i + 1

    if not new_embeddings:
        indexing_status.update({"running": False, "done": True, "phase": ""})
        return

    new_matrix = np.vstack(new_embeddings).astype('float32')

    if index is not None and len(photo_paths) > 0:
        existing_matrix = faiss.rev_swig_ptr(
            index.get_xb(), index.ntotal * index.d
        ).reshape(index.ntotal, index.d).copy()
        merged_matrix = np.vstack([existing_matrix, new_matrix]).astype('float32')
    else:
        merged_matrix = new_matrix

    merged_index = faiss.IndexFlatIP(merged_matrix.shape[1])
    merged_index.add(merged_matrix)
    merged_paths = list(photo_paths) + new_paths

    faiss.write_index(merged_index, INDEX_FILE_TMP)
    with open(PATHS_FILE_TMP, "wb") as f:
        pickle.dump(merged_paths, f)

    # Phase 2: Face Scanning
    indexing_status["phase"]    = "faces"
    indexing_status["progress"] = 0

    for i, path in enumerate(new_paths):
        try:
            detected = DeepFace.represent(
                path,
                model_name="ArcFace",
                enforce_detection=True,
                detector_backend="yunet"
            )
            valid_faces = []
            for d in detected:
                area = d.get("facial_area", {})
                w    = area.get("w", 0)
                h    = area.get("h", 0)
                emb  = np.array(d["embedding"], dtype=np.float32)
                if w >= 40 and h >= 40 and np.linalg.norm(emb) >= 2.0:
                    valid_faces.append(d["embedding"])
            if valid_faces:
                existing_faces[path] = valid_faces
        except:
            pass
        indexing_status["progress"] = i + 1

    with open(FACE_INDEX_FILE_TMP, "wb") as f:
        pickle.dump(existing_faces, f)

    os.replace(INDEX_FILE_TMP,      INDEX_FILE)
    os.replace(PATHS_FILE_TMP,      PATHS_FILE)
    os.replace(FACE_INDEX_FILE_TMP, FACE_INDEX_FILE)

    save_settings(folders)

    with index_lock:
        index             = merged_index
        photo_paths       = merged_paths
        _face_index_cache = existing_faces

    indexing_status.update({"running": False, "done": True, "phase": ""})

load_index()
_faces_cache      = load_faces()
_face_index_cache = load_face_index()

@app.post("/index")
def start_indexing(data: dict, background_tasks: BackgroundTasks):
    folders = data.get("folders", [])
    if not isinstance(folders, list) or not folders:
        return JSONResponse(content={"error": "Please provide a valid list of folders"}, status_code=400)
    for folder in folders:
        if not os.path.exists(folder.strip()):
            return JSONResponse(content={"error": f"Folder not found: {folder}"}, status_code=400)
    if indexing_status["running"]:
        return JSONResponse(content={"error": "The engine is already scanning folders right now"}, status_code=400)
    background_tasks.add_task(run_indexing, [f.strip() for f in folders])
    return {"message": "Folder scanning started"}

@app.post("/reindex")
def force_reindex(data: dict, background_tasks: BackgroundTasks):
    global index, photo_paths, _face_index_cache
    folders = data.get("folders", [])
    if not isinstance(folders, list) or not folders:
        return JSONResponse(content={"error": "Please provide a valid list of folders"}, status_code=400)
    if indexing_status["running"]:
        return JSONResponse(content={"error": "The engine is already scanning folders right now"}, status_code=400)
    index             = None
    photo_paths       = []
    _face_index_cache = {}
    for f in [INDEX_FILE, PATHS_FILE, FACE_INDEX_FILE]:
        if os.path.exists(f):
            os.remove(f)
    clean_folders = [f.strip() for f in folders if isinstance(f, str) and f.strip()]
    if not clean_folders:
        return JSONResponse(content={"error": "No valid folder paths provided"}, status_code=400)
    background_tasks.add_task(run_indexing, clean_folders)
    return {"message": "Database cleared. Fresh scan started"}

@app.get("/index-status")
def get_index_status():
    return indexing_status

@app.get("/settings")
def get_settings():
    return {"folders": load_settings()}

@app.get("/search")
def search(query: str, top_k: int = 40):
    with index_lock:
        current_index = index
        current_paths = list(photo_paths)

    if current_index is None:
        return JSONResponse(content={"error": "No photos have been scanned yet. Please map folders first"}, status_code=400)

    faces      = _faces_cache
    face_index = _face_index_cache
    clean_input = query.strip().lower()

    is_pure_name_search = False
    target_name = None
    for name in faces:
        pattern = r'^\s*' + re.escape(name.lower()) + r'\s*$'
        if re.match(pattern, clean_input):
            is_pure_name_search = True
            target_name = name
            break

    if is_pure_name_search:
        results = []
        if not face_index:
            return JSONResponse(content={"results": []})
        query_embeddings = faces[target_name]
        for path, embeddings in face_index.items():
            if not os.path.exists(path):
                continue
            try:
                best_score = 0.0
                for emb in embeddings:
                    score = face_similarity(query_embeddings, emb)
                    if score > best_score:
                        best_score = score
                if best_score >= 0.35:
                    results.append({
                        "filename": os.path.basename(path),
                        "path":     path,
                        "score":    float(best_score),
                        "url":      f"/file?path={path}"
                    })
            except:
                pass
        results.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse(content={"results": [] if not results else results})

    matched_name, clean_query = match_name_in_query(query, faces)
    tokens = tokenizer([clean_query if clean_query else query])
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vector    = text_features.cpu().numpy().astype('float32')
    actual_k        = min(len(current_paths), 100)
    scores, indices = current_index.search(query_vector, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(current_paths):
            continue
        path     = current_paths[idx]
        filename = os.path.basename(path)

        if matched_name:
            if path not in face_index:
                continue
            try:
                query_embeddings = faces[matched_name]
                best_score = 0.0
                for emb in face_index[path]:
                    s = face_similarity(query_embeddings, emb)
                    if s > best_score:
                        best_score = s
                if best_score >= 0.35:
                    results.append({
                        "filename": filename,
                        "path":     path,
                        "score":    float(score),
                        "url":      f"/file?path={path}"
                    })
            except:
                pass
        else:
            results.append({
                "filename": filename,
                "path":     path,
                "score":    float(score),
                "url":      f"/file?path={path}"
            })

    results.sort(key=lambda x: x["score"], reverse=True)

    seen_paths = set()
    unique_results = []
    for r in results:
        if r["path"] not in seen_paths:
            seen_paths.add(r["path"])
            unique_results.append(r)

    filtered = [r for r in unique_results if r["score"] >= 0.24]
    return JSONResponse(content={"results": filtered[:top_k]})

@app.post("/register-face")
async def register_face(data: dict):
    name      = data.get("name", "").strip()
    image_b64 = data.get("image", "")
    if not name or not image_b64:
        return JSONResponse(content={"error": "Missing name or reference photo"}, status_code=400)
    try:
        img_data   = base64.b64decode(image_b64)
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)
        
        timestamp_id = int(time.time() * 1000)
        img_filename = f"{timestamp_id}.jpg"
        img_path     = os.path.join(person_dir, img_filename)
        
        with open(img_path, "wb") as f:
            f.write(img_data)
        faces = load_faces()
        if name not in faces:
            faces[name] = []
        embedding = DeepFace.represent(
            img_path,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="yunet"
        )[0]["embedding"]
        emb_arr = np.array(embedding, dtype=np.float32)
        if np.linalg.norm(emb_arr) < 2.0:
            os.remove(img_path)
            return JSONResponse(content={"error": "Could not detect a clear face in the photo."}, status_code=400)
        faces[name].append(embedding)
        save_faces(faces)
        return {"message": f"Saved profile name entry for {name}", "total": len(faces[name])}
    except Exception as e:
        return JSONResponse(content={"error": "Failed to process profile photo. Make sure the face is clearly visible."}, status_code=400)

@app.get("/list-faces")
def list_faces():
    return {"people": [{"name": k, "photos": len(v)} for k, v in _faces_cache.items()]}

@app.delete("/delete-face")
def delete_face(name: str):
    faces = load_faces()
    if name in faces:
        del faces[name]
        save_faces(faces)
        person_dir = os.path.join(FACES_DIR, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
    return {"message": f"Deleted profile entry for: {name}"}

@app.get("/file")
def serve_file(path: str):
    allowed_folders = load_settings()
    if not is_path_safe(path, allowed_folders):
        return JSONResponse(content={"error": "Access denied"}, status_code=403)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Photo file not found on disk"}, status_code=404)
    res = FileResponse(path)
    res.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return res

@app.get("/download")
def download_file(path: str):
    allowed_folders = load_settings()
    if not is_path_safe(path, allowed_folders):
        return JSONResponse(content={"error": "Access denied"}, status_code=403)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Photo file not found on disk"}, status_code=404)
    return FileResponse(path, headers={"Content-Disposition": f"attachment; filename={os.path.basename(path)}"})

@app.get("/face-photo")
def get_face_photo(name: str, filename: str):
    img_path = os.path.normpath(os.path.join(FACES_DIR, name, filename))
    if not img_path.startswith(os.path.normpath(FACES_DIR)):
        return JSONResponse(content={"error": "Access Denied"}, status_code=403)
    if not os.path.exists(img_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    response = FileResponse(img_path)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response
      
@app.get("/face-photos")
def get_face_photos(name: str):
    person_dir = os.path.join(FACES_DIR, name)
    if not os.path.exists(person_dir):
        return JSONResponse(content={"error": "Not found"}, status_code=404)
    photos = sorted(os.listdir(person_dir))
    return {"photos": photos}

@app.get("/gallery-assets")
def get_gallery_assets():
    folders = load_settings()
    structured_data = {}
    for folder in folders:
        if os.path.exists(folder):
            structured_data[folder] = []
            try:
                all_files = []
                for root, dirs, files in os.walk(folder):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            full_path = os.path.join(root, f)
                            all_files.append((f, full_path))
                for f, full_path in sorted(all_files):
                    structured_data[folder].append({
                        "filename": f,
                        "path": full_path,
                        "url": f"/file?path={full_path}"
                    })
            except:
                pass
    return structured_data

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(APP_ROOT, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()