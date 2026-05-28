# Frontend — Loveable workflow

This folder is where the **React + Vite frontend** lives.

## Build flow

1. Open Loveable.
2. Paste **Section 1** from [`../LOVEABLE_PROMPT.md`](../LOVEABLE_PROMPT.md) as your first message.
3. As Loveable generates pages, use Sections 2–4 of that file to course-correct (especially Section 4 — Loveable will likely try to treat uploads as 2D images, which is wrong).
4. When the Loveable project looks right, export it (download zip) and unzip into this folder so the structure becomes:

   ```
   interface/frontend/
     package.json
     vite.config.ts
     index.html
     src/
       ...
     public/
       ...
   ```

5. From this folder:

   ```powershell
   npm install
   npm install @niivue/niivue              # Loveable may forget this
   npm run dev
   ```

6. Set the backend URL via env var (default is correct for local dev):

   ```powershell
   # interface/frontend/.env.local
   VITE_API_BASE_URL=http://localhost:8000
   ```

7. Once it builds, send me the folder tree (`tree /F src` or `Get-ChildItem -Recurse src`) and I'll wire the rough edges — typically:
   - axios base URL injection from `import.meta.env`
   - NiiVue integration (Loveable often stubs the canvas)
   - Session bootstrap (auto-create on first dashboard mount)
   - TanStack Query setup (queryClient + provider)

## Backend must be running

The frontend hits `http://localhost:8000` by default. Start the backend in another terminal:

```powershell
"C:\Users\oabde\miniconda3\envs\ai\python.exe" -m uvicorn interface.backend.main:app --reload --port 8000
```

If you change the backend port, update `VITE_API_BASE_URL` accordingly.

## Things to verify before declaring it done

- Upload a `.nii.gz` file via the dropzone — confirm a 201 from the backend and the new filename appears in the upload slot.
- NiiVue renders the volume after upload (not a black canvas or a 404).
- Pick `m2_t1c`, hit Run — metrics row populates and the overlay appears.
- Open `/compare`, pick `m2_t1c` + `m5_baseline`, hit Compare — both columns render with their own overlays and metrics.
- "Compare all" — 5 models populate (the 6th is correctly disabled with a "Coming soon" badge).
- Generate PDF report — file downloads, opens, and contains both metrics and slice images.
- Refresh the page — session id is restored from `localStorage` and the viewer reloads from the backend (this is the test that proves NiiVue isn't relying on the in-memory File object).
