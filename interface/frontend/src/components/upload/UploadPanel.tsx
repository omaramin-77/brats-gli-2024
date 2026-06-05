import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { UPLOAD_KINDS } from "@/lib/types";
import { UploadSlot } from "./UploadSlot";
import { useSession } from "@/store/useSession";
import { useCreateSession, useDeleteSession } from "@/api/queries";

export function UploadPanel() {
  const { sid, setSid, clearUploads } = useSession();
  const createSession = useCreateSession();
  const deleteSession = useDeleteSession();

  const reset = async () => {
    if (sid) {
      try { await deleteSession.mutateAsync(sid); } catch { /* ignore */ }
    }
    clearUploads();
    const fresh = await createSession.mutateAsync();
    setSid(fresh.sid);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="mb-3">
        <h2 className="text-sm font-semibold tracking-tight">Upload MRI</h2>
        <p className="text-xs text-muted-foreground">NIfTI volumes (.nii or .nii.gz)</p>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {UPLOAD_KINDS.map((k) => (
          <UploadSlot key={k} kind={k} />
        ))}
      </div>

      <Separator className="my-3" />
      <Button
        variant="outline"
        size="sm"
        onClick={reset}
        disabled={createSession.isPending || deleteSession.isPending}
      >
        <Trash2 className="h-3.5 w-3.5 mr-2" />
        Reset session
      </Button>
    </div>
  );
}
