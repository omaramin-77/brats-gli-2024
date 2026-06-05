export type UploadKind = "t1n" | "t1c" | "t2w" | "t2f" | "seg";

export const UPLOAD_KINDS: UploadKind[] = ["t1n", "t1c", "t2w", "t2f", "seg"];

export const MODALITY_LABELS: Record<UploadKind, { name: string; desc: string }> = {
  t1n: { name: "T1n", desc: "Native T1-weighted" },
  t1c: { name: "T1c", desc: "Contrast-enhanced T1" },
  t2w: { name: "T2w", desc: "T2-weighted" },
  t2f: { name: "T2f", desc: "T2-FLAIR" },
  seg: { name: "Seg", desc: "Ground-truth segmentation" },
};

export type ModelKey =
  | "m1_t1n"
  | "m2_t1c"
  | "m3_t2w"
  | "m4_t2f"
  | "m5_baseline"
  | "m6_late_fusion";

export type ModelInfo = {
  key: ModelKey;
  display_name: string;
  short_name: string;
  modality: "t1n" | "t1c" | "t2w" | "t2f" | "multimodal";
  in_channels: 1 | 4;
  architecture: string;
  description: string;
  badge: string;
  enabled: boolean;
  load_error: string | null;
};

export type ModelCatalog = { models: ModelInfo[] };

export type HealthResponse = {
  status: "ok";
  device: string;
  cuda_available: boolean;
  vram_gb: number;
  models_total: number;
  models_enabled: number;
  models_loaded: string[];
};

export type SessionInfo = {
  sid: string;
  created_at: number;
  updated_at: number;
  uploads: Record<string, string>;
  predictions: string[];
};

export type UploadAck = {
  sid: string;
  kind: string;
  filename: string;
  size_bytes: number;
};

export type PerSubregionMetrics = {
  model_key: string;
  model_display_name: string;
  model_short_name: string;
  architecture: string;
  modality: string;

  dice_wt: number; dice_tc: number; dice_et: number;
  iou_wt: number;  iou_tc: number;  iou_et: number;
  hd95_wt: number; hd95_tc: number; hd95_et: number;

  volume_mm3_wt: number; volume_mm3_tc: number; volume_mm3_et: number;
  inference_time_s: number;
};

export type ComparisonResponse = {
  sid: string;
  results: PerSubregionMetrics[];
  skipped: { model_key: string; reason: string }[];
};

export type Region = "wt" | "tc" | "et";
export type ViewAxis = "axial" | "coronal" | "sagittal";

export function modelRequiredModalities(m: ModelInfo): UploadKind[] {
  // Single-modality experts need their own modality + seg
  // Multimodal models need all 4 modalities + seg
  if (m.in_channels === 1 && m.modality !== "multimodal") {
    return [m.modality as UploadKind, "seg"];
  }
  return ["t1n", "t1c", "t2w", "t2f", "seg"];
}
