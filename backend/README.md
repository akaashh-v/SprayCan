# Precision Aerosol Spray Can Sound Detector — Backend API

Self-contained Flask + PyTorch Two-Stage Acoustic Event Detection backend service.

## 🚀 Hosting Options

Because the backend runs a deep spectro-temporal CNN (PyTorch) with digital signal processing (`scipy`, `soundfile`), it should be deployed to a container or Python service with adequate memory and CPU:
- **Render** (Recommended, Free tier / Individual)
- **Railway**
- **Fly.io**
- **Docker / AWS ECS / DigitalOcean**

---

### Option 1: Deploy on Render (Recommended)
1. Push this repository to GitHub.
2. In the Render Dashboard, click **New** > **Web Service**.
3. Select your repository.
4. Set **Root Directory**: `backend`.
5. Set **Environment**: `Python 3`.
6. Set **Build Command**:
   ```bash
   pip install --upgrade pip && pip install -r requirements.txt
   ```
7. Set **Start Command**:
   ```bash
   gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app
   ```
8. Add Environment Variables:
   - `CROSS_ORIGIN_COOKIES`: `true`
   - `FLASK_SECRET_KEY`: *(generate a random secure string)*
9. Click **Create Web Service**. Render will build and deploy your API with an HTTPS URL (e.g. `https://spray-can-detector-api.onrender.com`).

---

### Option 2: Deploy with Docker
Build and run the containerized backend anywhere:
```bash
cd backend
docker build -t spray-detector-backend .
docker run -p 5000:5000 -e PORT=5000 spray-detector-backend
```

---

### Option 3: Deploy on Railway
1. Click **New Project** > **Deploy from GitHub repo**.
2. Set Root Directory to `backend`.
3. Railway automatically detects `backend/Procfile` or `backend/Dockerfile` and deploys.

---

## 🔒 CORS & Security
The backend automatically permits cross-origin requests from any frontend domain, supports preflight `OPTIONS` requests, and configures `SameSite=None` secure cookies for cross-origin authentication.
