import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Region, UploadKind, ViewAxis } from "@/lib/types";

type ViewerSettings = {
  backdrop: UploadKind;
  view: ViewAxis;
  slice: number;
  opacity: number;
  regions: Record<Region, boolean>;
};

export type StagedUpload = { filename: string; size: number } | null;

type State = {
  sid: string | null;
  uploads: Record<UploadKind, StagedUpload>;
  selectedModelKey: string | null;
  compareSet: string[];
  viewerSettings: ViewerSettings;
  activePredictionKey: string | null;

  setSid: (sid: string | null) => void;
  setUpload: (kind: UploadKind, info: StagedUpload) => void;
  clearUploads: () => void;
  setSelectedModelKey: (k: string | null) => void;
  toggleCompare: (k: string) => void;
  setCompareSet: (keys: string[]) => void;
  updateViewer: (patch: Partial<ViewerSettings>) => void;
  setActivePrediction: (k: string | null) => void;
};

const initialUploads: Record<UploadKind, StagedUpload> = {
  t1n: null, t1c: null, t2w: null, t2f: null, seg: null,
};

export const useSession = create<State>()(
  persist(
    (set) => ({
      sid: null,
      uploads: initialUploads,
      selectedModelKey: null,
      compareSet: [],
      activePredictionKey: null,
      viewerSettings: {
        backdrop: "t1c",
        view: "axial",
        slice: 64,
        opacity: 0.5,
        regions: { wt: true, tc: true, et: true },
      },
      setSid: (sid) => set({ sid }),
      setUpload: (kind, info) =>
        set((s) => ({ uploads: { ...s.uploads, [kind]: info } })),
      clearUploads: () => set({ uploads: initialUploads, activePredictionKey: null }),
      setSelectedModelKey: (k) => set({ selectedModelKey: k }),
      toggleCompare: (k) =>
        set((s) =>
          s.compareSet.includes(k)
            ? { compareSet: s.compareSet.filter((x) => x !== k) }
            : { compareSet: [...s.compareSet, k] },
        ),
      setCompareSet: (keys) => set({ compareSet: keys }),
      updateViewer: (patch) =>
        set((s) => ({ viewerSettings: { ...s.viewerSettings, ...patch } })),
      setActivePrediction: (k) => set({ activePredictionKey: k }),
    }),
    {
      name: "brats-session",
      partialize: (s) => ({ sid: s.sid, uploads: s.uploads }),
    },
  ),
);
