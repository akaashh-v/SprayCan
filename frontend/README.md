# Precision Aerosol Spray Can Sound Detector — Frontend

A decoupled, static Web Audio API frontend dashboard for the Precision Aerosol Spray Can Sound Detector.

## 🚀 Hosting Options

This frontend is purely static (HTML5, Vanilla CSS3, Modern JavaScript) and can be hosted on any static hosting platform for free:
- **Vercel**
- **Netlify**
- **Cloudflare Pages**
- **GitHub Pages**

---

### Option 1: Deploy on Vercel (Recommended)
1. Push this repository to GitHub/GitLab.
2. In the Vercel Dashboard, click **New Project** and select your repository.
3. In **Root Directory**, click **Edit** and select: `frontend`.
4. Click **Deploy**. Vercel will automatically use `frontend/vercel.json` and serve the dashboard.

---

### Option 2: Deploy on Netlify
1. In Netlify, click **Add new site** > **Import an existing project**.
2. Set **Base directory**: `frontend`.
3. Set **Publish directory**: `frontend`.
4. Click **Deploy site**.

---

## 🌐 Connecting to the Backend API

When deployed to a separate domain (e.g. `https://spray-can.vercel.app`), connect it to your live backend (e.g. on Render, Railway, or Fly.io):

1. Click the **🌐 API URL** button in the top navbar.
2. Enter your backend URL, for example:
   ```
   https://spray-can-detector-api.onrender.com
   ```
3. The page will reload and all acoustic analysis, file evaluation, and telemetry calls will seamlessly communicate with your hosted backend over secure CORS.
4. *(To reset back to local development, click **API URL** again, leave the field empty, and press OK).*
