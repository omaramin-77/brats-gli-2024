import { useRef, useState } from "react";
import { Check, Upload, X, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useSession } from "@/store/useSession";
import { useUpload } from "@/api/queries";
import type { UploadKind } from "@/lib/types";
import { MODALITY_LABELS } from "@/lib/types";
import { cn } from "@/lib/utils";

export function UploadSlot({ kind }: { kind: UploadKind }) {
  const { uploads, setUpload, sid } = useSession();
  const staged = uploads[kind];
  const upload = useUpload(sid);
  const [pct, setPct] = useState(0);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const label = MODALITY_LABELS[kind];
  const accentClass =
    kind === "seg" ? "border-l-accent" : "border-l-primary";

  const handleFiles = async (files: FileList | null) => {
    if (!files || !files[0]) return;
    const file = files[0];
    if (!file.name.endsWith(".nii") && !file.name.endsWith(".nii.gz")) {
      return;
    }
    setPct(0);
    try {
      const ack = await upload.mutateAsync({
        kind,
        file,
        onProgress: setPct,
      });
      setUpload(kind, { filename: ack.filename, size: ack.size_bytes });
    } catch {
      /* toast handled in mutation */
    }
  };

  const clear = () => {
    setUpload(kind, null);
    setPct(0);
  };

  return (
    <Card
      className={cn(
        "border-l-4 p-3 transition-colors",
        accentClass,
        drag && "ring-2 ring-primary/40",
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{label.name}</span>
            {staged && <Check className="h-3.5 w-3.5 text-success" />}
          </div>
          <p className="text-xs text-muted-foreground">{label.desc}</p>
        </div>
        {staged && (
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={clear}>
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {!staged ? (
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className="cursor-pointer rounded-md border border-dashed border-border bg-muted/30 px-3 py-4 text-center text-xs text-muted-foreground hover:bg-muted/60 transition-colors"
        >
          {upload.isPending ? (
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <Progress value={pct} className="h-1" />
              <span className="font-mono">{pct}%</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-1.5">
              <Upload className="h-4 w-4" />
              <span>Drop .nii / .nii.gz</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".nii,.nii.gz,application/gzip"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>
      ) : (
        <div className="rounded-md bg-muted/40 px-2.5 py-2 font-mono text-[11px]">
          <div className="truncate">{staged.filename}</div>
          <div className="text-muted-foreground">{(staged.size / 1024 / 1024).toFixed(2)} MB</div>
        </div>
      )}
    </Card>
  );
}
