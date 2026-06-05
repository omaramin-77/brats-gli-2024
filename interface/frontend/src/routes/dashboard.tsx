import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Copy, FileDown, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/Header";
import { UploadPanel } from "@/components/upload/UploadPanel";
import { NiiVueCanvas } from "@/components/viewer/NiiVueCanvas";
import { ViewerToolbar } from "@/components/viewer/ViewerToolbar";
import { ModelCard, ModelDownloadLink } from "@/components/models/ModelCard";
import { MetricsTable } from "@/components/metrics/MetricsTable";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/store/useSession";
import {
  useCreateSession,
  useModels,
  usePrediction,
  useRunPrediction,
  useSessionInfo,
  useCompare,
  useCompareAll,
} from "@/api/queries";
import type { UploadKind } from "@/lib/types";
import { reportUrl } from "@/api/client";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard — BraTS GLI-2024" },
      { name: "description", content: "Upload MRI volumes and run tumour-segmentation models." },
    ],
  }),
  component: Dashboard,
});

function Dashboard() {
  const navigate = useNavigate();
  const {
    sid, setSid, uploads, selectedModelKey, setSelectedModelKey,
    compareSet, toggleCompare, viewerSettings, activePredictionKey,
    setActivePrediction,
  } = useSession();

  const createSession = useCreateSession();
  const { data: models } = useModels();
  const { data: serverSession } = useSessionInfo(sid);
  const runPrediction = useRunPrediction(sid);
  const compare = useCompare(sid);
  const compareAll = useCompareAll(sid);
  const { data: currentMetrics } = usePrediction(sid, activePredictionKey);

  const [sliceMax, setSliceMax] = useState(155);

  // Auto-create session on mount
  useEffect(() => {
    if (!sid && !createSession.isPending) {
      createSession.mutate(undefined, {
        onSuccess: (s) => setSid(s.sid),
      });
    }
  }, [sid, createSession.isPending]);

  const uploadedSet = useMemo<Set<UploadKind>>(() => {
    const s = new Set<UploadKind>();
    (Object.keys(uploads) as UploadKind[]).forEach((k) => {
      if (uploads[k]) s.add(k);
    });
    return s;
  }, [uploads]);

  const mriUploaded: UploadKind[] = (["t1n", "t1c", "t2w", "t2f"] as UploadKind[])
    .filter((k) => uploadedSet.has(k));

  const backdropAvailable =
    viewerSettings.backdrop && uploadedSet.has(viewerSettings.backdrop)
      ? viewerSettings.backdrop
      : mriUploaded[0] ?? null;

  const predictionsArr = serverSession?.predictions ?? [];

  const handleRun = (modelKey: string) => {
    setSelectedModelKey(modelKey);
    runPrediction.mutate(modelKey, {
      onSuccess: () => setActivePrediction(modelKey),
    });
  };

  const handleCompareSelected = async () => {
    if (compareSet.length < 2) {
      toast.error("Select at least 2 models");
      return;
    }
    try {
      await compare.mutateAsync(compareSet);
      navigate({
        to: "/compare",
        search: { sid: sid!, models: compareSet.join(",") },
      });
    } catch { /* toast handled */ }
  };

  const handleCompareAll = async () => {
    try {
      const res = await compareAll.mutateAsync();
      const keys = res.results.map((r) => r.model_key);
      navigate({
        to: "/compare",
        search: { sid: sid!, models: keys.join(",") },
      });
    } catch { /* toast handled */ }
  };

  const copySid = () => {
    if (!sid) return;
    navigator.clipboard.writeText(sid);
    toast.success("Session ID copied");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="border-b border-border bg-card/40">
        <div className="mx-auto max-w-[1600px] px-6 py-3 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-semibold">Analysis dashboard</h1>
            {sid && (
              <button
                onClick={copySid}
                className="flex items-center gap-1.5 rounded-md border border-border bg-muted/40 px-2.5 py-1 font-mono text-[11px] text-muted-foreground hover:text-foreground"
              >
                {sid.slice(0, 8)}…<Copy className="h-3 w-3" />
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <a
              href={sid && predictionsArr.length ? reportUrl(sid, predictionsArr) : undefined}
              target="_blank"
              rel="noreferrer"
              aria-disabled={!predictionsArr.length}
            >
              <Button
                size="sm"
                variant="outline"
                disabled={!sid || !predictionsArr.length}
              >
                <FileDown className="h-3.5 w-3.5 mr-2" />
                Generate PDF report
              </Button>
            </a>
          </div>
        </div>
      </div>

      <main className="flex-1 mx-auto max-w-[1600px] w-full px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left — Upload */}
          <aside className="lg:col-span-3">
            <Card className="p-4 h-full max-h-[calc(100vh-12rem)]">
              <UploadPanel />
            </Card>
          </aside>

          {/* Center — Viewer */}
          <section className="lg:col-span-6 flex flex-col gap-3">
            <ViewerToolbar
              available={mriUploaded}
              sliceMax={sliceMax}
              hasOverlay={!!activePredictionKey}
            />
            <Card className="p-2 flex-1 min-h-[520px]">
              {backdropAvailable ? (
                <NiiVueCanvas
                  sid={sid}
                  backdrop={backdropAvailable}
                  hasBackdropUploaded={true}
                  overlayModelKey={activePredictionKey}
                  view={viewerSettings.view}
                  opacity={viewerSettings.opacity}
                  regions={viewerSettings.regions}
                  onSliceMax={setSliceMax}
                />
              ) : (
                <div className="h-full min-h-[480px] grid place-items-center text-center text-sm text-muted-foreground">
                  <div>
                    <div className="mb-2 font-mono text-xs uppercase tracking-wider">No volume loaded</div>
                    <div>Upload at least one MRI modality to preview</div>
                  </div>
                </div>
              )}
            </Card>
            {activePredictionKey && (
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  Overlay: <span className="font-mono text-foreground">{activePredictionKey}</span>
                </span>
                {sid && <ModelDownloadLink sid={sid} modelKey={activePredictionKey} />}
              </div>
            )}
          </section>

          {/* Right — Models + metrics */}
          <aside className="lg:col-span-3 flex flex-col">
            <Tabs defaultValue="single" className="flex-1 flex flex-col">
              <TabsList className="w-full">
                <TabsTrigger value="single" className="flex-1">Single model</TabsTrigger>
                <TabsTrigger value="compare" className="flex-1">Compare</TabsTrigger>
              </TabsList>

              <TabsContent value="single" className="mt-3 space-y-2 flex-1 overflow-y-auto pr-1">
                {models?.models.map((m) => (
                  <ModelCard
                    key={m.key}
                    model={m}
                    uploaded={uploadedSet}
                    mode="single"
                    selected={selectedModelKey === m.key}
                    running={runPrediction.isPending && runPrediction.variables === m.key}
                    hasPrediction={predictionsArr.includes(m.key)}
                    onRun={() => handleRun(m.key)}
                    onSelect={() => {
                      setSelectedModelKey(m.key);
                      if (predictionsArr.includes(m.key)) setActivePrediction(m.key);
                    }}
                  />
                ))}

                {currentMetrics && (
                  <>
                    <Separator className="my-3" />
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm font-semibold">{currentMetrics.model_short_name}</div>
                          <Badge variant="outline" className="font-mono text-[10px]">
                            {currentMetrics.architecture}
                          </Badge>
                        </div>
                        <span className="font-mono text-xs text-muted-foreground">
                          {currentMetrics.inference_time_s.toFixed(2)}s
                        </span>
                      </div>
                      <MetricsTable m={currentMetrics} />
                      {sid && <ModelDownloadLink sid={sid} modelKey={currentMetrics.model_key} />}
                    </div>
                  </>
                )}
              </TabsContent>

              <TabsContent value="compare" className="mt-3 space-y-2 flex-1 overflow-y-auto pr-1">
                <div className="flex gap-2 mb-2">
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={handleCompareSelected}
                    disabled={compareSet.length < 2 || compare.isPending}
                  >
                    {compare.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : `Compare (${compareSet.length})`}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1"
                    onClick={handleCompareAll}
                    disabled={compareAll.isPending}
                  >
                    {compareAll.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Compare all"}
                  </Button>
                </div>
                {models?.models.map((m) => (
                  <ModelCard
                    key={m.key}
                    model={m}
                    uploaded={uploadedSet}
                    mode="compare"
                    selected={compareSet.includes(m.key)}
                    running={false}
                    hasPrediction={predictionsArr.includes(m.key)}
                    onRun={() => handleRun(m.key)}
                    onSelect={() => toggleCompare(m.key)}
                    onToggleCompare={() => toggleCompare(m.key)}
                  />
                ))}
              </TabsContent>
            </Tabs>
          </aside>
        </div>
      </main>
    </div>
  );
}

// silence unused
void Link;
