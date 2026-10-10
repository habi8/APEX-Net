# APEX-Net

Backend code is in `DacNet/`. Frontend code is in
https://github.com/habi8/apex-net-frontend.

## Run locally (Windows PowerShell)

### 1. Start the backend

The API runs from `DacNet/` and loads the APEX-Net classifier, lung U-Net,
and all 14 disease prior maps when it starts. Make sure these files exist
before launching it:

- `DacNet/models/ngp6gwkn/best_model_20261006-050300.pth`
- `DacNet/models/lung_unet.pth`
- `DacNet/prior_maps/<Disease>.npy` (one file for each of the 14 findings)

The repository's default paths can be changed with the environment variables
listed below. The following setup uses the CPU-only PyTorch wheels; the first
installation can take a while.

From the repository root, open a PowerShell terminal and run:

```powershell
Set-Location .\DacNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install -r api-requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Wait for all three model resources to load. In another terminal, confirm
`http://127.0.0.1:8000/health` returns `"status": "ok"`; interactive API
documentation is at `http://127.0.0.1:8000/docs`.

To test a prediction from PowerShell, replace the image path and run:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "file=@C:\path\to\chest-xray.png" `
  -F "heatmap_count=3"
```

The response contains findings with sigmoid model scores and class-specific
heatmap overlays. The scores are not calibrated diagnostic probabilities.
The API also accepts an optional `X-API-Key` header if `APEX_API_KEY` is set
in the backend process environment before startup:

```powershell
$env:APEX_API_KEY = "your-local-development-key"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

When API-key protection is enabled, include the same key in the prediction
request with `-H "X-API-Key: your-local-development-key"`. Do not commit
real keys to the repository.

Optional backend environment variables:

- `APEX_CHECKPOINT`: APEX-Net classifier checkpoint.
- `LUNG_CHECKPOINT`: lung U-Net checkpoint.
- `PRIOR_MAP_DIR`: directory containing the 14 `.npy` disease prior maps.
- `APEX_DEVICE`: `cpu`, `cuda`, or `mps`; by default CUDA is used when
  available, otherwise CPU.
- `FRONTEND_ORIGINS`: comma-separated browser origins allowed by CORS;
  defaults to `http://localhost:3000,http://127.0.0.1:3000`.
- `APEX_API_KEY`: optional API key required by `/predict` when configured.

### 2. Start the frontend

In a second terminal, go to the frontend directory (the folder containing
`package.json`; for a sibling checkout, `..\APEX-Net-frontend\apex-net-frontend`):

```powershell
npm install
npm run dev
```

Create `.env.local` in that directory with your Supabase
`NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`. The frontend
forwards predictions to `http://127.0.0.1:8000` by default. If you enable
backend API-key protection, set `APEX_API_KEY` for the backend and set the
same value as `APEX_BACKEND_API_KEY` in the frontend `.env.local`.

Open http://localhost:3000.
