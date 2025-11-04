from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import base64
import ollama
import re
from fastapi import Request
import os
import uuid
import tempfile
import open3d as o3d
from sklearn.neighbors import KDTree
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2, io, numpy as np
import ollama
# Load model and scaler with error handling
try:
    model = tf.keras.models.load_model("rockfall_model (1).keras")
    scaler = joblib.load("scaler (1).pkl")
    try:
        max_values = pd.read_csv("m (1).csv", index_col=0, on_bad_lines="skip").squeeze("columns")
    except Exception as e:
        print(f"Error loading max_values CSV: {e}")
        max_values = None
except FileNotFoundError as e:
    print(f"Error: Required model or data file not found: {e}")
    model = None
    scaler = None
    max_values = None

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Input schema
# -------------------------
class InputData(BaseModel):
    depth_m: float
    moisture_percent: float
    slope_angle_deg: float
    rock_density_gcm3: float
    seismic_zone: int
    distance_from_water_m: float
    pore_pressure_kPa: float
    zone: str

# Store history in memory
history = []

def preprocess_input(data):
    if max_values is None:
        raise HTTPException(status_code=500, detail="Maximum values data not loaded. Check server logs.")
    max_arr = max_values.values
    scaled = data / max_arr
    return scaled

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model not loaded. Check server logs.")
    try:
        X = np.array([[data.depth_m,
                       data.moisture_percent,
                       data.slope_angle_deg,
                       data.rock_density_gcm3,
                       data.seismic_zone,
                       data.distance_from_water_m,
                       data.pore_pressure_kPa]])
        X_scaled = preprocess_input(X)
        prob = model.predict(X_scaled)[0][0]
        formatted_prob = f"{prob:.2f}"
        result = {
            "zone": data.zone,
            "depth_m": data.depth_m,
            "moisture_percent": data.moisture_percent,
            "slope_angle_deg": data.slope_angle_deg,
            "rock_density_gcm3": data.rock_density_gcm3,
            "seismic_zone": data.seismic_zone,
            "distance_from_water_m": data.distance_from_water_m,
            "pore_pressure_kPa": data.pore_pressure_kPa,
            "risk_prob": float(formatted_prob)
        }
        history.append(result)

        return {"risk_prob": float(formatted_prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze_and_predict")
async def analyze_and_predict(request: Request):
    try:
        # ✅ Parse JSON body
        data = await request.json()
        data_obj = InputData(**data)

        # === Step 1: ML risk prediction ===
        prob = 0.0
        if model:
            try:
                X = np.array([[data_obj.depth_m,
                               data_obj.moisture_percent,
                               data_obj.slope_angle_deg,
                               data_obj.rock_density_gcm3,
                               data_obj.seismic_zone,
                               data_obj.distance_from_water_m,
                               data_obj.pore_pressure_kPa]])
                X_scaled = preprocess_input(X)
                prob = model.predict(X_scaled)[0][0]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

        # === Step 2: Build prompt for Ollama ===
        text_prompt = f"""
        Current conditions:
        - Zone: {data_obj.zone}
        - Depth: {data_obj.depth_m} m
        - Moisture: {data_obj.moisture_percent} %
        - Slope Angle: {data_obj.slope_angle_deg} °
        - Rock Density: {data_obj.rock_density_gcm3} g/cm³
        - Seismic Zone: {data_obj.seismic_zone}
        - Distance from Water: {data_obj.distance_from_water_m} m
        - Pore Pressure: {data_obj.pore_pressure_kPa} kPa
        - Predicted Risk Probability: {prob:.2f}

        Provide a structured safety protocol with sections:
        - Risk Assessment
        - Preventive Actions
        - Work Sequence
        - Additional Recommendations
        """

        # === Step 3: Call Ollama ===
        try:
            response = ollama.chat(
                model='rockfall-analyst',
                messages=[{'role': 'user', 'content': text_prompt}]
            )
            protocol_text = response['message']['content']
        except Exception as e:
            protocol_text = f"⚠️ Ollama call failed: {e}"

        # === Step 4: Return results ===
        return {
            "risk_prob": float(prob),
            "safety_protocol": protocol_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/")
async def analyze_data_post(data: InputData):
    # ... (code to prepare the prompt)
    
    # Send the request to Ollama
    try:
        response = ollama.chat(
            model='rockfall-analyst:latest',
            messages=[{'role': 'user', 'content': prompt}],
            options={'stop': ['<|im_end|>']}
        )
        llm_response = response['message']['content']
        
        # Return the raw text directly
        return {"report": llm_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")
    
@app.get("/history/{zone}")
def get_history(zone: str):
    """Get past prediction history for a specific zone"""
    zone_history = [h for h in history if h.get("zone") == zone]
    return zone_history

# Make a static folder for saving processed files
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static folder if not already
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except Exception:
    pass


# ---- Risk calculation helpers ----
def compute_local_slope_and_curvature(points, k=30):
    """
    Compute local slope (degrees) and curvature for each point.
    points: (N,3) numpy array
    """
    tree = KDTree(points)
    neigh_idx = tree.query(points, k=k, return_distance=False)

    slopes = np.zeros(len(points))
    curvatures = np.zeros(len(points))

    for i, idxs in enumerate(neigh_idx):
        pts = points[idxs] - points[i]
        C = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        normal = eigvecs[:, 0]
        if normal[2] < 0:
            normal = -normal
        slope_rad = np.arccos(np.clip(normal[2], -1.0, 1.0))
        slopes[i] = np.degrees(slope_rad)
        curvatures[i] = eigvals[0] / (eigvals.sum() + 1e-12)

    return slopes, curvatures


def risk_from_geometry(slope_deg, curvature):
    """Combine slope + curvature into a risk value [0..1]."""
    slope_comp = np.clip((slope_deg - 15) / (60 - 15), 0, 1)
    curv_comp = np.clip(curvature * 10, 0, 1)
    return np.clip(0.75 * slope_comp + 0.25 * curv_comp, 0, 1)


def risk_to_rgb(risk):
    """Map risk [0..1] → RGB color (green→yellow→red)."""
    r = np.zeros_like(risk)
    g = np.zeros_like(risk)
    b = np.zeros_like(risk)

    low = risk <= 0.5
    t = risk[low] / 0.5
    r[low] = t
    g[low] = 1
    b[low] = 0

    high = risk > 0.5
    t2 = (risk[high] - 0.5) / 0.5
    r[high] = 1
    g[high] = 1 - t2
    b[high] = 0

    return np.stack([r, g, b], axis=1)


# ---- API Endpoints ----
@app.post("/upload_pointcloud/")
async def upload_pointcloud(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext != ".ply":
            return JSONResponse({"error": "Only .ply files supported"}, status_code=400)

        uid = uuid.uuid4().hex
        tmp_path = os.path.join(tempfile.gettempdir(), f"{uid}.ply")
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        pcd = o3d.io.read_point_cloud(tmp_path)
        if pcd.is_empty():
            return JSONResponse({"error": "Empty point cloud"}, status_code=400)

        points = np.asarray(pcd.points)

        if len(points) > 200000:
            pcd = pcd.voxel_down_sample(voxel_size=0.1)
            points = np.asarray(pcd.points)

        # ✅ Convert to mesh using Ball Pivoting or Poisson reconstruction
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Try Poisson surface reconstruction for smooth mesh
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Compute slope/curvature on mesh vertices
        vertices = np.asarray(mesh.vertices)
        slopes, curvs = compute_local_slope_and_curvature(vertices, k=30)
        risks = risk_from_geometry(slopes, curvs)
        colors = risk_to_rgb(risks)
        
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        out_name = f"processed_{uid}.ply"
        out_path = os.path.join(STATIC_DIR, out_name)
        o3d.io.write_triangle_mesh(out_path, mesh)  # ✅ Save as mesh, not point cloud

        return {
            "filename": file.filename,
            "points": len(vertices),
            "slope_mean": float(np.mean(slopes)),
            "slope_max": float(np.max(slopes)),
            "processed_url": f"/static/{out_name}"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    


@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    """Analyze uploaded image for cracks and moisture traces."""
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            return JSONResponse({"error": "Only .jpg/.jpeg/.png allowed"}, status_code=400)

        uid = uuid.uuid4().hex
        temp_path = os.path.join(tempfile.gettempdir(), f"{uid}{ext}")

        # ✅ ensure the file is fully written
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        file.file.close()

        # --- Image analysis ---
        image = cv2.imread(temp_path)
        if image is None:
            raise ValueError(f"Unable to read image at {temp_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        crack_score = np.mean(edges) / 255 * 100

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_moist = np.array([90, 30, 30])
        upper_moist = np.array([140, 255, 255])
        mask_moist = cv2.inRange(hsv, lower_moist, upper_moist)
        moisture_score = np.mean(mask_moist) / 255 * 100

        overlay = image.copy()
        overlay[edges > 100] = [0, 0, 255]
        overlay[mask_moist > 0] = [255, 0, 0]

        out_name = f"analyzed_{uid}.jpg"
        out_path = os.path.join(STATIC_DIR, out_name)
        cv2.imwrite(out_path, overlay)

        # --- AI interpretation ---
        ai_report = "Analysis not available."
        try:
            prompt = f"""
            The analyzed mine surface image has:
            - Crack intensity: {crack_score:.2f}%
            - Moisture presence: {moisture_score:.2f}%

            Provide a short safety assessment and next recommended actions.
            """

            response = ollama.chat(
                model='rockfall-analyst',
                messages=[{"role": "user", "content": prompt}]
            )
            ai_report = response["message"]["content"]
        except Exception as e:
            ai_report = f"Ollama interpretation failed: {str(e)}"

        return {
            "filename": file.filename,
            "crack_intensity_percent": round(crack_score, 2),
            "moisture_presence_percent": round(moisture_score, 2),
            "processed_url": f"/static/{out_name}",
            "ai_report": ai_report
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
