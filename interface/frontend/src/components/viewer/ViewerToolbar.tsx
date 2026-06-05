import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { Region, UploadKind, ViewAxis } from "@/lib/types";
import { useSession } from "@/store/useSession";

type Props = {
  available: UploadKind[];
  sliceMax: number;
  hasOverlay: boolean;
};

const VIEWS: ViewAxis[] = ["axial", "coronal", "sagittal"];

const REGION_LABELS: Record<Region, { label: string; cls: string }> = {
  wt: { label: "WT", cls: "bg-region-wt" },
  tc: { label: "TC", cls: "bg-region-tc" },
  et: { label: "ET", cls: "bg-region-et" },
};

export function ViewerToolbar({ available, sliceMax, hasOverlay }: Props) {
  const { viewerSettings: v, updateViewer } = useSession();
  const mriOptions: UploadKind[] = available.filter((k) => k !== "seg");

  return (
    <div className="space-y-3 rounded-md border border-border bg-card/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <RadioGroup
          value={v.backdrop}
          onValueChange={(val) => updateViewer({ backdrop: val as UploadKind })}
          className="flex gap-3"
        >
          {(["t1n", "t1c", "t2w", "t2f"] as UploadKind[]).map((k) => {
            const enabled = mriOptions.includes(k);
            return (
              <div key={k} className="flex items-center gap-1.5">
                <RadioGroupItem id={`m-${k}`} value={k} disabled={!enabled} />
                <Label
                  htmlFor={`m-${k}`}
                  className={`text-xs font-mono uppercase ${enabled ? "" : "opacity-40"}`}
                >
                  {k}
                </Label>
              </div>
            );
          })}
        </RadioGroup>

        <Tabs value={v.view} onValueChange={(val) => updateViewer({ view: val as ViewAxis })}>
          <TabsList className="h-8">
            {VIEWS.map((vw) => (
              <TabsTrigger key={vw} value={vw} className="text-xs capitalize px-3">
                {vw}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <div className="flex items-center gap-3">
        <Label className="text-xs text-muted-foreground w-12">Slice</Label>
        <Slider
          value={[v.slice]}
          min={0}
          max={Math.max(sliceMax - 1, 1)}
          step={1}
          onValueChange={([val]) => updateViewer({ slice: val })}
          className="flex-1"
        />
        <span className="font-mono text-xs w-12 text-right">{v.slice}/{sliceMax}</span>
      </div>

      {hasOverlay && (
        <div className="space-y-2 border-t border-border pt-2">
          <div className="flex items-center gap-3">
            <Label className="text-xs text-muted-foreground w-12">Opacity</Label>
            <Slider
              value={[v.opacity * 100]}
              min={0}
              max={100}
              step={5}
              onValueChange={([val]) => updateViewer({ opacity: val / 100 })}
              className="flex-1"
            />
            <span className="font-mono text-xs w-12 text-right">{Math.round(v.opacity * 100)}%</span>
          </div>
          <div className="flex flex-wrap gap-3">
            {(Object.keys(REGION_LABELS) as Region[]).map((r) => (
              <div key={r} className="flex items-center gap-2">
                <span className={`h-3 w-3 rounded-sm ${REGION_LABELS[r].cls}`} />
                <Label className="text-xs">{REGION_LABELS[r].label}</Label>
                <Switch
                  checked={v.regions[r]}
                  onCheckedChange={(c) =>
                    updateViewer({ regions: { ...v.regions, [r]: c } })
                  }
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
