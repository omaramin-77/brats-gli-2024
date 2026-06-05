import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, Loader2 } from "lucide-react";
import { z } from "zod";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis,
} from "recharts";
import { Header } from "@/components/layout/Header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { NiiVueCanvas } from "@/components/viewer/NiiVueCanvas";
import { useSession } from "@/store/useSession";
import { useCompare } from "@/api/queries";
import type { PerSubregionMetrics, UploadKind } from "@/lib/types";

const searchSchema = z.object({
  sid: z.string().optional(),
  models: z.string().optional(),
});

export const Route = createFileRoute("/compare")({
  validateSearch: (s) => searchSchema.parse(s),
  head: () => ({
    meta: [
      { title: "Compare — BraTS GLI-2024" },
      { name: "description", content: "Side-by-side comparison of segmentation models." },
    ],
  }),
  component: ComparePage,
});

function fmt(n: number, d = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  return n.toFixed(d);
}

function ComparePage() {
  const { sid: searchSid, models: modelsParam } = Route.useSearch();
  const { sid: storeSid, uploads, viewerSettings } = useSession();
  const sid = searchSid ?? storeSid;
  const keys = useMemo(() => (modelsParam ? modelsParam.split(",").filter(Boolean) : []), [modelsParam]);

  const compare = useCompare(sid);
  const [results, setResults] = useState<PerSubregionMetrics[]>([]);
  const [sortKey, setSortKey] = useState<string>("model");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  useEffect(() => {
    if (!sid || keys.length < 2) return;
    compare.mutate(keys, {
      onSuccess: (r) => setResults(r.results),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sid, modelsParam]);

  const backdrop: UploadKind = uploads.t1c ? "t1c" : (Object.keys(uploads).find((k) => uploads[k as UploadKind]) as UploadKind) ?? "t1c";

  const bestValues = useMemo(() => {
    const cols = ["dice_wt","dice_tc","dice_et","iou_wt","iou_tc","iou_et"];
    const out: Record<string, number> = {};
    for (const c of cols) {
      out[c] = Math.max(...results.map((r) => (r as any)[c] ?? -Infinity));
    }
    const hd = ["hd95_wt","hd95_tc","hd95_et"];
    for (const c of hd) {
      const vals = results.map((r) => (r as any)[c]).filter((v) => !Number.isNaN(v) && v !== null);
      out[c] = vals.length ? Math.min(...vals) : NaN;
    }
    return out;
  }, [results]);

  const sorted = useMemo(() => {
    if (sortKey === "model") return results;
    return [...results].sort((a, b) => {
      const av = (a as any)[sortKey] ?? 0;
      const bv = (b as any)[sortKey] ?? 0;
      return sortDir === "asc" ? av - bv : bv - av;
    });
  }, [results, sortKey, sortDir]);

  const chartData = useMemo(
    () => results.map((r) => ({
      name: r.model_short_name,
      WT: r.dice_wt, TC: r.dice_tc, ET: r.dice_et,
    })),
    [results],
  );

  if (!sid) {
    return (
      <div className="min-h-screen flex flex-col">
        <Header />
        <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
          No session — <Link to="/dashboard" className="ml-1 text-primary underline">go to dashboard</Link>
        </div>
      </div>
    );
  }

  const isLoading = compare.isPending || results.length === 0;

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 mx-auto max-w-[1600px] w-full px-6 py-6">
        <div className="flex items-center justify-between mb-5">
          <div>
            <Link to="/dashboard" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground mb-2">
              <ArrowLeft className="h-3.5 w-3.5" />
              Back to dashboard
            </Link>
            <h1 className="text-2xl font-bold tracking-tight">Model comparison</h1>
            <p className="text-sm text-muted-foreground mt-1">
              {keys.length} models · session <span className="font-mono">{sid.slice(0, 8)}</span>
            </p>
          </div>
        </div>

        <Tabs defaultValue="grid">
          <TabsList>
            <TabsTrigger value="grid">Grid</TabsTrigger>
            <TabsTrigger value="table">Metrics table</TabsTrigger>
          </TabsList>

          <TabsContent value="grid" className="mt-4">
            {isLoading ? (
              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
                {keys.map((k: string) => (
                  <Card key={k} className="p-4 h-80 animate-pulse">
                    <div className="h-4 w-32 bg-muted rounded mb-2" />
                    <div className="h-3 w-24 bg-muted rounded mb-4" />
                    <div className="h-48 bg-muted/50 rounded" />
                  </Card>
                ))}
              </div>
            ) : (
              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
                {sorted.map((r) => (
                  <Card key={r.model_key} className="p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <div className="font-semibold">{r.model_short_name}</div>
                        <Badge variant="outline" className="font-mono text-[10px] mt-1">{r.architecture}</Badge>
                      </div>
                      <span className="font-mono text-xs text-muted-foreground">{r.inference_time_s.toFixed(2)}s</span>
                    </div>
                    <div className="h-56 mb-3 rounded-md overflow-hidden">
                      <NiiVueCanvas
                        sid={sid}
                        backdrop={backdrop}
                        hasBackdropUploaded={!!uploads[backdrop]}
                        overlayModelKey={r.model_key}
                        view={viewerSettings.view}
                        opacity={0.55}
                        regions={{ wt: true, tc: true, et: true }}
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-2 font-mono text-xs">
                      {(["wt","tc","et"] as const).map((reg) => (
                        <div key={reg} className="rounded-sm border border-border px-2 py-1.5">
                          <div className={`text-[10px] uppercase mb-0.5 text-region-${reg}`}>Dice {reg}</div>
                          <div className="text-sm leading-relaxed">{fmt((r as any)[`dice_${reg}`])}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="table" className="mt-4 space-y-6">
            {isLoading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                <Card className="overflow-hidden">
                  <div className="overflow-x-auto">
                    <table className="w-full font-mono text-xs leading-relaxed">
                      <thead className="bg-muted/40">
                        <tr>
                          {[
                            ["model","Model"],
                            ["dice_wt","Dice WT"],["dice_tc","Dice TC"],["dice_et","Dice ET"],
                            ["iou_wt","IoU WT"],["iou_tc","IoU TC"],["iou_et","IoU ET"],
                            ["hd95_wt","HD95 WT"],["hd95_tc","HD95 TC"],["hd95_et","HD95 ET"],
                            ["inference_time_s","Time s"],
                          ].map(([k, label]) => (
                            <th
                              key={k}
                              onClick={() => {
                                if (sortKey === k) setSortDir(sortDir === "asc" ? "desc" : "asc");
                                else { setSortKey(k); setSortDir(k === "model" ? "asc" : "desc"); }
                              }}
                              className="cursor-pointer px-3 py-2 font-medium text-left whitespace-nowrap hover:bg-muted/70"
                            >
                              {label}{sortKey === k ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sorted.map((r) => (
                          <tr key={r.model_key} className="border-t border-border">
                            <td className="px-3 py-2">{r.model_short_name}</td>
                            {(["dice_wt","dice_tc","dice_et","iou_wt","iou_tc","iou_et"] as const).map((c) => {
                              const v = (r as any)[c];
                              const best = bestValues[c] === v && !Number.isNaN(v);
                              return (
                                <td key={c} className={`px-3 py-2 text-right ${best ? "text-accent font-semibold" : ""}`}>
                                  {fmt(v)}
                                </td>
                              );
                            })}
                            {(["hd95_wt","hd95_tc","hd95_et"] as const).map((c) => {
                              const v = (r as any)[c];
                              const best = bestValues[c] === v && !Number.isNaN(v);
                              return (
                                <td key={c} className={`px-3 py-2 text-right ${best ? "text-accent font-semibold" : ""}`}>
                                  {fmt(v, 2)}
                                </td>
                              );
                            })}
                            <td className="px-3 py-2 text-right">{r.inference_time_s.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                <Card className="p-4">
                  <h3 className="text-sm font-semibold mb-3">Dice by region</h3>
                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                        <RTooltip
                          contentStyle={{
                            backgroundColor: "var(--popover)",
                            border: "1px solid var(--border)",
                            borderRadius: 6,
                            fontSize: 12,
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="WT" fill="#32b4dc" />
                        <Bar dataKey="TC" fill="#ffa500" />
                        <Bar dataKey="ET" fill="#dc323c" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
