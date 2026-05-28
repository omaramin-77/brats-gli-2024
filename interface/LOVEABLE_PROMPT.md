# Loveable Prompt — BraTS Brain MRI Tumour Segmentation Frontend

> **How to use this file**
> Paste the entire **Section 1** below into Loveable as your first message. Sections 2–4 are reference: keep them to one side and paste the relevant block when Loveable asks a follow-up question or you need to nudge a specific page. The corrections in Section 4 are the ones it will most likely need.

---

# SECTION 1 — Paste this into Loveable

You are building the frontend for **Brain MRI Tumour Segmentation — BraTS GLI-2024**, a research dashboard that runs deep-learning segmentation models on **3D MRI volumes**. The FastAPI backend is already built. Your job is the React/Vite frontend that talks to it.

## 1. Critical constraints — do not deviate

1. **The data is 3D NIfTI volumes (`.nii.gz`), not 2D images.** Do not use generic image upload / `<img>` previews. Upload accepts only `.nii` and `.nii.gz` files (~5–30 MB each). The MRI viewer is **NiiVue** — the WebGL NIfTI viewer at <https://github.com/niivue/niivue>. Install with `npm i @niivue/niivue`. Use it everywhere you'd otherwise render an image.
2. **There are five upload slots per case**: `t1n`, `t1c`, `t2w`, `t2f`, and `seg` (ground-truth segmentation, always available in this project's dataset). Each upload is a separate NIfTI file. The slots must be visually distinct.
3. **API endpoints and request/response shapes are fixed.** Do not invent endpoints or fields. Copy the contract in section 4 exactly. Use a base URL of `http://localhost:8000` for development, overridable via `VITE_API_BASE_URL`.
4. **There are 6 models**, identified by these exact `key` strings: `m1_t1n`, `m2_t1c`, `m3_t2w`, `m4_t2f`, `m5_baseline`, `m6_late_fusion`. The 6th is intentionally `enabled: false` for now — show it with a "Coming soon" badge.
5. **Single-modality models only need their own modality.** Do not require the user to upload all 4 modalities before they can run `m2_t1c`. The backend will return a `400` with a `missing: [...]` list if uploads are insufficient.
6. **The metrics are real Dice/IoU/HD95 against ground truth.** The backend computes them server-side and returns them in the prediction response. Display them; do not fabricate them.

## 2. Tech stack

- **React 18 + Vite + TypeScript**
- **TailwindCSS** with `shadcn/ui` components (button, card, dialog, tabs, tooltip, toast, progress, select, slider, badge, separator, sheet, table)
- **Framer Motion** for the landing-page hero animation and page transitions only — no decorative motion on data-heavy panels
- **NiiVue** (`@niivue/niivue`) for the MRI viewer
- **TanStack Query** (`@tanstack/react-query`) for all server state
- **Zustand** for client-side UI state only (session id, selected models, active view) — not for server data
- **TanStack Router** (file-based routing under `src/routes/`) — accept Loveable's default if it offers TanStack Router; do not migrate to react-router-dom. Run in SPA mode, not SSR — there is no Node backend.
- **lucide-react** for icons
- **axios** for HTTP (configure a single instance with `VITE_API_BASE_URL`)

Do not use Redux, MobX, or Recoil. Do not add a state management library other than Zustand.

## 3. Design system

This is a medical research tool aimed at a graduation-defense audience. The aesthetic is **modern, minimal, medical AI** — closer to Hugging Face Spaces and Linear than to a hospital EHR.

- **Palette** — dark-mode-first with a clean light variant:
  - Background `#0a0f1c` (dark) / `#fafbfc` (light)
  - Surface `#111827` (dark) / `#ffffff` (light)
  - Primary `#3b82f6` (medical-blue)
  - Accent `#22d3ee` (cyan — matches BraTS WT colour)
  - Tumour-region colours (frozen — these match the backend's overlay PNGs):
    - WT — `#32b4dc` cyan-blue
    - TC — `#ffa500` orange
    - ET — `#dc323c` crimson
  - Success `#10b981`, warning `#f59e0b`, danger `#ef4444`
- **Typography**: Inter for body, JetBrains Mono for metrics tables and file names
- **Density**: Spacious but not sparse. 16px base, 24px section padding, generous line-height on numbers (`leading-relaxed`)
- **Corners**: `rounded-xl` for cards, `rounded-md` for inputs/buttons. Avoid `rounded-full` except on avatars/badges
- **Shadows**: subtle — `shadow-sm` for cards, no glow/neon
- **Glassmorphism**: optional on the landing hero only. Never on data panels
- **Theme toggle** in the top-right header. Default to dark on first load

## 4. Pages and routing

```
/                      Landing page
/dashboard             Analysis dashboard
/compare               Comparison view
/about                 About / how it works (static)
```

### 4.1 Landing page (`/`)
- Top nav with project title left, theme toggle + GitHub link + "Start Analysis" CTA right
- Hero section: project title, one-paragraph explanation, animated MRI-glyph illustration (use Framer Motion to softly cycle through 4 NIfTI-style cross-sections — placeholder NiiVue with a built-in demo volume is fine, or an animated SVG of layered ellipses if NiiVue feels heavy here)
- Three info cards in a row explaining the model families: **Modality experts (4)**, **Naive Baseline**, **Late Fusion (coming soon)**. Pull descriptions from `GET /models` at load time so the cards are always in sync with the backend.
- Footer with a disclaimer: "Research prototype — not for clinical use. Trained on BraTS GLI-2024."

### 4.2 Dashboard (`/dashboard`)

Three-column layout on desktop (`lg:grid-cols-12`), stacked on mobile. The layout MUST persist across uploads — do not navigate away during inference.

**Left column (3 cols)** — Upload panel:
- Heading "Upload MRI"
- Five upload slots stacked vertically: T1n, T1c, T2w, T2f, Seg (ground truth). Each slot is a `Card` with:
  - The modality name + a one-line description ("Native T1", "Contrast-enhanced T1", etc.)
  - A drag-and-drop zone (accepts only `.nii`/`.nii.gz`)
  - When a file is staged: filename, size, an "X" to clear
  - Upload progress (use TanStack Query mutation `onUploadProgress`)
  - Filled slot shows a green check icon. Missing slot shows a grey placeholder
- A "Reset session" button at the bottom that deletes the session via `DELETE /sessions/{sid}` and creates a new one

**Center column (6 cols)** — NIfTI viewer:
- A NiiVue canvas that fills the column. Show a NiiVue toolbar with:
  - Modality selector — radio group across the 4 uploaded MRI modalities (greys out missing)
  - View selector — Axial / Coronal / Sagittal tabs
  - Slice slider — scrubs through Z (or Y/X depending on view)
  - Overlay toggle — once a prediction is loaded, toggle WT/TC/ET overlay; per-region opacity slider
  - "Before / After" comparison — a horizontal slider that reveals the overlay over the underlying MRI in a wipe pattern (see Loveable's image-comparison-slider pattern, but for the WebGL viewer)
- When no modality is uploaded: empty-state with a hint "Upload at least one modality to preview"
- Use NiiVue's `addVolumeFromUrl` — point it at `GET /sessions/{sid}/uploads/{kind}` for the base modality and `GET /sessions/{sid}/predictions/{model_key}/mask.nii.gz` for the overlay. The frontend should **never** parse NIfTI in JS itself.

**Right column (3 cols)** — Model picker + metrics:
- Tab strip: "Single model" / "Compare"
- "Single model" tab: a vertical list of all 6 models from `GET /models`. Each row shows:
  - Display name + badge ("Modality expert" / "Baseline" / "Coming soon")
  - One-line description
  - Required modalities — small chips. Each chip is green if uploaded, grey if missing
  - A "Run inference" button — disabled with a tooltip when:
    - Model is `enabled: false`, OR
    - Any required modality is missing
- After clicking Run: button shows a spinner; on success, scroll the right column to show the metrics panel below, and the viewer overlays the new mask automatically
- Metrics panel — a table with three rows (WT, TC, ET) and four columns (Dice, IoU, HD95, Volume mm³). Numbers right-aligned, monospace. HD95 shows "n/a" if NaN. Below: "Inference time: X.XX s" and a "Download mask (NIfTI)" link
- "Compare" tab: identical model list but with **multi-select checkboxes**. Two buttons: "Compare selected" (requires ≥ 2) and "Compare all" (runs every enabled model)

**Header (above the 3 columns)**:
- Session id badge (truncated, click to copy)
- "Generate PDF report" button (only enabled after ≥ 1 prediction exists)
- Theme toggle

### 4.3 Comparison view (`/compare?sid=...&models=...`)

Reached by clicking "Compare selected" or "Compare all". Two view modes (tabs):

**Mode 1 — Grid**:
- A responsive grid of cards, one per model. Each card has:
  - Model name + architecture
  - A NiiVue canvas with the same underlying T1c modality + that model's predicted overlay
  - A compact metrics row: Dice WT / TC / ET
  - Inference time
- 1 column on mobile, 2 on tablet, 3 on desktop (so "Compare all" lays out 2×3)

**Mode 2 — Metrics table**:
- Sortable table. Columns: Model, Dice WT, Dice TC, Dice ET, IoU WT/TC/ET, HD95 WT/TC/ET, Inference time
- The best value per column highlighted in cyan
- Below the table: a small bar chart (Dice per region per model, grouped) — use `recharts`

A "Back to dashboard" button restores the dashboard with the same session.

### 4.4 About (`/about`)
- Static page explaining the 4 modalities (with example illustrations), the WT/TC/ET sub-regions, and a 3-bullet summary of how each model architecture differs. Pull from `GET /models` for the architecture names.
- Disclaimer reiterated at the bottom

## 5. State management

**Zustand store** (`useSession`) — client-only:
- `sid: string | null`
- `uploads: Record<UploadKind, { filename: string; size: number } | null>` for the 5 kinds
- `selectedModelKey: string | null` (single-mode)
- `compareSet: Set<string>` (compare-mode)
- `viewerSettings: { backdrop, view, slice, opacity, region }`

**TanStack Query** — all server data, including:
- `useModels()` — `GET /models`, stale time 1 hour
- `useSession(sid)` — `GET /sessions/{sid}`, refetch after every upload
- `useHealth()` — `GET /health`, 30s interval (small device + cuda indicator in the header footer)
- `usePrediction(sid, key)` — derives from `GET /sessions/{sid}/predictions/{key}/metrics`, populated by the `useRunPrediction` mutation

**Persistence**: store `sid` in `localStorage` so a page reload doesn't lose the in-flight analysis.

## 6. API integration

All endpoints are on the FastAPI backend (`http://localhost:8000` by default). Wrap them in `src/api/client.ts` as typed functions returning Promises. Use TanStack Query everywhere. **Every shape below is exact — do not add, remove, or rename fields.**

### 6.1 `GET /health`

Response:
```ts
type HealthResponse = {
  status: "ok";
  device: string;                // e.g. "cuda" or "cpu"
  cuda_available: boolean;
  vram_gb: number;
  models_total: number;
  models_enabled: number;
  models_loaded: string[];
};
```

### 6.2 `GET /models`

Response:
```ts
type ModelInfo = {
  key: "m1_t1n" | "m2_t1c" | "m3_t2w" | "m4_t2f" | "m5_baseline" | "m6_late_fusion";
  display_name: string;
  short_name: string;
  modality: "t1n" | "t1c" | "t2w" | "t2f" | "multimodal";
  in_channels: 1 | 4;
  architecture: string;
  description: string;
  badge: string;                 // "Modality expert" | "Baseline" | "Coming soon"
  enabled: boolean;
  load_error: string | null;
};

type ModelCatalog = { models: ModelInfo[] };
```

`in_channels === 1` means the model only needs **its own** modality (plus `seg`). `in_channels === 4` means the model needs all four modalities (plus `seg`).

### 6.3 `POST /sessions`

Body: none. Response:
```ts
type SessionInfo = {
  sid: string;
  created_at: number;            // unix seconds
  updated_at: number;
  uploads: Record<string, string>;   // kind -> stored filename
  predictions: string[];             // model keys
};
```

### 6.4 `POST /sessions/{sid}/upload`

`multipart/form-data` with two fields:
- `kind`: one of `"t1n" | "t1c" | "t2w" | "t2f" | "seg"`
- `file`: the `.nii` or `.nii.gz` File object

Response:
```ts
type UploadAck = {
  sid: string;
  kind: string;
  filename: string;
  size_bytes: number;
};
```

### 6.5 `POST /sessions/{sid}/predict`

Body:
```ts
{ model_key: string }
```

Response:
```ts
type PerSubregionMetrics = {
  model_key: string;
  model_display_name: string;
  model_short_name: string;
  architecture: string;
  modality: string;

  dice_wt: number; dice_tc: number; dice_et: number;
  iou_wt: number;  iou_tc: number;  iou_et: number;
  hd95_wt: number; hd95_tc: number; hd95_et: number;    // may be NaN

  volume_mm3_wt: number; volume_mm3_tc: number; volume_mm3_et: number;
  inference_time_s: number;
};
```

Errors:
- `400` with `{ detail: "missing uploads for m2_t1c: ['t1c', 'seg']..." }` — missing modalities
- `503` with `{ detail: "model m6_late_fusion not available: ..." }` — model not enabled

### 6.6 `POST /sessions/{sid}/compare` and `/compare-all`

Body for `/compare`:
```ts
{ model_keys: string[] }   // at least 2
```

Response (both endpoints):
```ts
type ComparisonResponse = {
  sid: string;
  results: PerSubregionMetrics[];
  skipped: { model_key: string; reason: string }[];
};
```

### 6.7 NIfTI downloads — what NiiVue consumes

These endpoints return raw `.nii.gz` files. Feed their URLs directly into NiiVue via `addVolumeFromUrl`. **The frontend must never parse NIfTI in JavaScript itself.**

- `GET /sessions/{sid}/uploads/{kind}` — raw uploaded modality (`kind` ∈ `t1n | t1c | t2w | t2f | seg`). This is the base volume the viewer renders. Without it, NiiVue cannot render anything after a page refresh.
- `GET /sessions/{sid}/predictions/{model_key}/mask.nii.gz` — the 4D prediction mask (channels = WT, TC, ET). Add this as a NiiVue overlay on top of the base volume.
- `GET /sessions/{sid}/predictions/{model_key}/metrics` — JSON; same shape as `PerSubregionMetrics`.
- `GET /sessions/{sid}/predictions/{model_key}/slice?view=axial&index=64&region=all&backdrop=t1c` — returns a PNG. **Only use this as a fallback** if NiiVue overlay is unavailable; the canvas-based viewer is the primary path.

### 6.8 Report and cleanup

- `GET /sessions/{sid}/report.pdf?models=key1,key2` — triggers a PDF download. Render as a normal `<a download>` link with the URL.
- `DELETE /sessions/{sid}` — call on "Reset session"; returns 204.

## 7. Important behavior rules

1. **Auto-create a session** on first mount of `/dashboard` if `sid` is null in the Zustand store.
2. **Re-fetch session metadata** after every successful upload so the model-picker rows update their "required modalities" chips.
3. **Show a single global toast** on every API error using the `detail` field from the response — backend always returns `{ detail: "..." }` for errors.
4. **NiiVue overlay management**: when a prediction succeeds, add the mask NIfTI as a second volume with `opacity = 0.5`. When the user switches predictions, remove the old overlay before adding the new one (don't stack).
5. **The viewer's "region" toggle**: show WT, TC, ET each as a separate NiiVue overlay layer that the user can toggle independently. Use the colours from §3.
6. **Inference time**: render with 2 decimal places and a "s" suffix. Models can take 5–60 seconds depending on device — show a clear progress indicator (indeterminate is fine; do not fake a percentage).
7. **Compare-all** runs models sequentially server-side. Render skeleton rows for each model and fill them in as `results` arrives. If the response is one large blob (no streaming), still display them ordered by the catalog order, not by speed.

## 8. Folder structure (target)

```
src/
  api/
    client.ts              axios instance + endpoint functions, typed
    queries.ts             all TanStack Query hooks
  components/
    layout/                Header, ThemeToggle, PageShell
    upload/                UploadSlot, UploadPanel
    viewer/                NiiVueCanvas, ViewerToolbar, RegionTogglePanel, BeforeAfterSlider
    models/                ModelCard, ModelPicker, CompareModeToggle
    metrics/               MetricsTable, MetricsBarChart
    common/                ErrorBoundary, EmptyState, KbdBadge
  routes/                  TanStack Router file-based routes
    __root.tsx             root layout (Header + Outlet)
    index.tsx              /         → LandingPage
    dashboard.tsx          /dashboard → DashboardPage
    compare.tsx            /compare   → ComparePage (reads sid + models from search params)
    about.tsx              /about     → AboutPage
  store/
    useSession.ts          Zustand
  styles/
    globals.css            Tailwind + theme tokens
  lib/
    nifti.ts               small NiiVue helpers (build URL, set colormap)
    types.ts               TS types matching §6
  routeTree.gen.ts         auto-generated by TanStack Router — do not edit
  main.tsx
```

## 9. What NOT to do

- Do **not** use `<img>` previews or thumbnail grids — there are no 2D images in this project
- Do **not** generate fake metrics or fake inference progress percentages
- Do **not** add user authentication / login pages — this is single-user
- Do **not** add a backend; one exists at `http://localhost:8000`
- Do **not** invent endpoints not in §6
- Do **not** use websockets — the backend is HTTP-only
- Do **not** add Tailwind plugins beyond what shadcn requires
- Do **not** use emojis in the UI or in code

---

# SECTION 2 — When Loveable asks "what should the hero look like"

Show a left-aligned bold title "Brain MRI Tumour Segmentation", a 2-sentence subtitle ("Deep-learning sub-region segmentation across four MRI modalities. Trained on BraTS GLI-2024 — research use only."), a primary "Start Analysis" button, and to the right a NiiVue canvas slowly auto-rotating through a demo volume. Below, a 3-card row summarising the model families. Subtle gradient background. No carousel, no testimonials, no pricing.

---

# SECTION 3 — When Loveable asks "what should empty states look like"

- **No session, dashboard loaded**: large NiiVue canvas with a neutral colour, centered icon (`lucide-react`'s `UploadCloud`), text "Upload an MRI modality to begin", a single primary button labelled "Pick file" that opens the OS file dialog targeting the T1c slot.
- **Session exists, no uploads**: dim the viewer column, highlight the T1c slot on the left with a glow ring, tooltip "Start with T1c — it has the clearest tumour signal".
- **Uploads done, no predictions**: highlight the "Run inference" button on a recommended model (M2 T1c if T1c is uploaded, else M5 baseline if all 4 are uploaded).

---

# SECTION 4 — Corrections to apply if Loveable goes off-script

Paste any of these into Loveable to course-correct.

### "Treat all uploads as 3D NIfTI volumes, not images"

> The uploads are 3D NIfTI volumes (.nii.gz files containing a (H, W, D) float array plus an affine matrix). They are not images. Replace any `<img>` or image-thumbnail rendering with a NiiVue canvas (`@niivue/niivue`). NiiVue accepts a NIfTI File or URL and renders the volume; the user scrubs through slices via NiiVue's built-in controls. There is no 2D preview anywhere in this app.

### "Use the exact API contract"

> All API endpoints are listed in the original prompt under §6. Do not invent endpoints. Do not rename fields. Do not assume there is a `/users`, `/auth`, `/me`, or `/projects` endpoint — none exist. The backend is single-tenant and stateless apart from sessions. Errors always return `{ detail: string }`.

### "Use NiiVue for the viewer, not a 2D image overlay"

> The MRI viewer must use the `@niivue/niivue` package. Initialise a `Niivue` instance, attach it to a canvas via `attachToCanvas`, then call `addVolumeFromUrl` for the base modality and again for the prediction mask. Switching modality means remove + re-add the base volume. NiiVue handles slice scrubbing, orientation, and overlay opacity natively — do not reimplement them in React.

### "Drag-and-drop must accept only NIfTI files"

> The dropzone's `accept` is `.nii,.nii.gz` and only these extensions. Reject everything else with a toast. Maximum file size 200 MB. Each of the 5 slots is independent — uploading T1c does not stage T1n.

### "Do not require all 4 modalities for single-modality models"

> Models with `in_channels === 1` only need their own modality plus the seg ground-truth. The model picker should look up `required_modalities` for each model:
> - `m1_t1n` → `["t1n", "seg"]`
> - `m2_t1c` → `["t1c", "seg"]`
> - `m3_t2w` → `["t2w", "seg"]`
> - `m4_t2f` → `["t2f", "seg"]`
> - `m5_baseline` → `["t1n", "t1c", "t2w", "t2f", "seg"]`
> - `m6_late_fusion` → same as m5_baseline (when enabled)
> Disable the "Run inference" button only when one of the required-for-this-model uploads is missing.

### "Metrics come from the backend — display them, don't compute them"

> Every prediction response includes dice_wt, dice_tc, dice_et, iou_wt/tc/et, hd95_wt/tc/et, volume_mm3_wt/tc/et, and inference_time_s. The frontend's job is to render these in a table. Do not implement any segmentation metric in JS. Treat NaN values for hd95 as "n/a".

---

# After Loveable finishes

Drop the generated project under `interface/frontend/`. Send me the folder tree and I'll wire any rough edges: the `axios` client base URL via env var, the NiiVue integration if it generated a placeholder, and the session bootstrap flow.
