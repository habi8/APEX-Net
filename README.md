# APEX-Net

Backend code is in `DacNet/`. Frontend code is in
https://github.com/habi8/apex-net-frontend.

## Run locally (Windows PowerShell)

### 1. Start the backend

From the repository root:

```powershell
cd DacNet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
python -m pip install -r api-requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Wait for the models to load, then confirm the API at
http://localhost:8000/health reports `"status": "ok"`.

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
