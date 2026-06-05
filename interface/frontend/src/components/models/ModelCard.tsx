import { Loader2, Play, Download } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import type { ModelInfo, UploadKind } from "@/lib/types";
import { modelRequiredModalities } from "@/lib/types";
import { useSession } from "@/store/useSession";

type Props = {
  model: ModelInfo;
  uploaded: Set<UploadKind>;
  mode: "single" | "compare";
  selected: boolean;
  running: boolean;
  hasPrediction: boolean;
  onRun: () => void;
  onSelect: () => void;
  onToggleCompare?: () => void;
};

const BADGE_VARIANT: Record<string, "default" | "secondary" | "outline"> = {
  "Modality expert": "default",
  Baseline: "secondary",
  "Coming soon": "outline",
};

export function ModelCard({
  model,
  uploaded,
  mode,
  selected,
  running,
  hasPrediction,
  onRun,
  onSelect,
  onToggleCompare,
}: Props) {
  const required = modelRequiredModalities(model);
  const missing = required.filter((r) => !uploaded.has(r));
  const disabled = !model.enabled || missing.length > 0;

  const disabledReason = !model.enabled
    ? model.load_error ?? "Model not enabled"
    : missing.length > 0
      ? `Missing: ${missing.join(", ")}`
      : "";

  return (
    <Card
      className={cn(
        "p-3 cursor-pointer transition-all border",
        selected ? "border-primary ring-1 ring-primary/40" : "hover:border-primary/40",
        !model.enabled && "opacity-60",
      )}
      onClick={mode === "single" ? onSelect : onToggleCompare}
    >
      <div className="flex items-start justify-between gap-2 mb-1.5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {mode === "compare" && (
              <Checkbox
                checked={selected}
                onCheckedChange={onToggleCompare}
                disabled={!model.enabled}
                onClick={(e) => e.stopPropagation()}
              />
            )}
            <span className="font-semibold text-sm truncate">{model.short_name}</span>
            {hasPrediction && <span className="h-1.5 w-1.5 rounded-full bg-success" />}
          </div>
          <p className="text-xs text-muted-foreground line-clamp-1">{model.architecture}</p>
        </div>
        <Badge variant={BADGE_VARIANT[model.badge] ?? "outline"} className="shrink-0 text-[10px]">
          {model.badge}
        </Badge>
      </div>

      <p className="text-xs text-muted-foreground line-clamp-2 mb-2">{model.description}</p>

      <div className="flex flex-wrap gap-1 mb-2">
        {required.map((r) => (
          <span
            key={r}
            className={cn(
              "rounded-sm px-1.5 py-0.5 font-mono text-[10px] uppercase",
              uploaded.has(r) ? "bg-success/15 text-success" : "bg-muted text-muted-foreground",
            )}
          >
            {r}
          </span>
        ))}
      </div>

      {mode === "single" && (
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="block">
              <Button
                size="sm"
                className="w-full h-8"
                disabled={disabled || running}
                onClick={(e) => {
                  e.stopPropagation();
                  onRun();
                }}
              >
                {running ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5 mr-1.5" />
                    Run inference
                  </>
                )}
              </Button>
            </span>
          </TooltipTrigger>
          {disabled && disabledReason && (
            <TooltipContent>{disabledReason}</TooltipContent>
          )}
        </Tooltip>
      )}
    </Card>
  );
}

export function ModelDownloadLink({ sid, modelKey }: { sid: string; modelKey: string }) {
  const url = `${(import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000"}/sessions/${sid}/predictions/${modelKey}/mask.nii.gz`;
  return (
    <a href={url} download className="inline-flex items-center gap-1.5 text-xs text-primary hover:underline">
      <Download className="h-3.5 w-3.5" />
      Download mask (NIfTI)
    </a>
  );
}

export { useSession };
