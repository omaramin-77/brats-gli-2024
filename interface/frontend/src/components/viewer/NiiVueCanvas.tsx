import { useEffect, useRef } from "react";
import type { Region, UploadKind, ViewAxis } from "@/lib/types";
import { maskUrl, uploadUrl } from "@/api/client";

type Props = {
  sid: string | null;
  backdrop: UploadKind | null;
  hasBackdropUploaded: boolean;
  overlayModelKey?: string | null;
  view: ViewAxis;
  opacity: number;
  regions: Record<Region, boolean>;
  onSliceMax?: (max: number) => void;
};

const REGION_COLORS: Record<Region, string> = {
  wt: "#32b4dc",
  tc: "#ffa500",
  et: "#dc323c",
};

const VIEW_MAP: Record<ViewAxis, number> = {
  axial: 2,
  coronal: 1,
  sagittal: 0,
};

export function NiiVueCanvas(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<any>(null);
  const lastBackdropRef = useRef<string | null>(null);
  const lastOverlayRef = useRef<string | null>(null);

  // Init NiiVue once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const mod = await import("@niivue/niivue");
      if (cancelled || !canvasRef.current) return;
      const nv = new mod.Niivue({
        backColor: [0.04, 0.06, 0.11, 1],
        show3Dcrosshair: false,
        textHeight: 0.03,
        crosshairColor: [0.23, 0.51, 0.96, 0.8],
      });
      await nv.attachToCanvas(canvasRef.current);
      nv.setSliceType(nv.sliceTypeAxial);
      nvRef.current = nv;
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Load backdrop volume
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !props.sid || !props.backdrop || !props.hasBackdropUploaded) return;
    const url = uploadUrl(props.sid, props.backdrop);
    if (url === lastBackdropRef.current) return;
    lastBackdropRef.current = url;
    lastOverlayRef.current = null;
    (async () => {
      try {
        while (nv.volumes.length) nv.removeVolumeByIndex(0);
        await nv.loadVolumes([{ url, colormap: "gray", opacity: 1 }]);
        const dims = nv.volumes[0]?.dims ?? [0, 0, 0, 0];
        const axis = VIEW_MAP[props.view];
        const max = dims[axis + 1] ?? 128;
        props.onSliceMax?.(max);
      } catch (e) {
        console.error("NiiVue load error", e);
      }
    })();
  }, [props.sid, props.backdrop, props.hasBackdropUploaded]);

  // Update slice axis
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv) return;
    const map = {
      axial: nv.sliceTypeAxial,
      coronal: nv.sliceTypeCoronal,
      sagittal: nv.sliceTypeSagittal,
    };
    nv.setSliceType(map[props.view]);
  }, [props.view]);

  // Manage overlay
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !props.sid) return;

    // Remove existing overlays (everything except the first volume)
    while (nv.volumes.length > 1) nv.removeVolumeByIndex(nv.volumes.length - 1);

    if (!props.overlayModelKey) {
      lastOverlayRef.current = null;
      nv.updateGLVolume();
      return;
    }

    const url = maskUrl(props.sid, props.overlayModelKey);
    (async () => {
      try {
        // Add 3 overlay layers for WT / TC / ET. The mask is 4D — niivue exposes channels via "frame".
        // We add the same URL multiple times with different colormaps and frames.
        const enabledRegions: Region[] = (["wt", "tc", "et"] as Region[]).filter(
          (r) => props.regions[r],
        );
        for (let i = 0; i < enabledRegions.length; i++) {
          const region = enabledRegions[i];
          await nv.addVolumeFromUrl({
            url,
            name: `mask_${region}`,
            colormap: "red",
            opacity: props.opacity,
            cal_min: 0.5,
            cal_max: 1,
            frame4D: ["wt", "tc", "et"].indexOf(region),
          });
          const v = nv.volumes[nv.volumes.length - 1];
          if (v) {
            const hex = REGION_COLORS[region];
            const rgb = hexToRgb(hex);
            // Build a 1-color colormap
            v.colormapLabel = null;
            v.colormap = "";
            v.colormapNegative = "";
            v.colorMap = "";
            // Use a custom colormap
            const cmap = {
              R: [0, rgb[0]],
              G: [0, rgb[1]],
              B: [0, rgb[2]],
              A: [0, 255],
              I: [0, 1],
            };
            try {
              nv.addColormap(`region_${region}`, cmap);
              v.colormap = `region_${region}`;
            } catch {
              /* colormap may already exist */
              v.colormap = `region_${region}`;
            }
          }
        }
        lastOverlayRef.current = url;
        nv.updateGLVolume();
      } catch (e) {
        console.error("Overlay load error", e);
      }
    })();
  }, [props.sid, props.overlayModelKey, props.opacity, props.regions.wt, props.regions.tc, props.regions.et]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full h-full block rounded-md bg-[#0a0f1c]"
    />
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}
