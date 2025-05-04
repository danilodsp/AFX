"""
REST API prototype using FastAPI (for future extension).
"""
from fastapi import FastAPI, UploadFile, File
from AFX.io.io import load_audio
from AFX.utils.config_loader import load_config
from AFX.extract_all import extract_all_features
import tempfile

app = FastAPI()

@app.post('/extract')
async def extract_features(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    config = load_config('audio_features/config.json')
    signal, sr = load_audio(tmp_path, sr=config['sample_rate'])
    features = extract_all_features(signal, sr, config)
    return {k: v.tolist() for k, v in features.items()}
