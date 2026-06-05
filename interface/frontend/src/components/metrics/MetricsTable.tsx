import type { PerSubregionMetrics, Region } from "@/lib/types";

const REGIONS: Region[] = ["wt", "tc", "et"];
const REGION_LABELS: Record<Region, string> = { wt: "WT", tc: "TC", et: "ET" };

function fmtNum(n: number, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  return n.toFixed(digits);
}

function fmtVol(n: number): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "n/a";
  return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function MetricsTable({ m }: { m: PerSubregionMetrics }) {
  return (
    <div className="overflow-hidden rounded-md border border-border">
      <table className="w-full font-mono text-xs leading-relaxed">
        <thead className="bg-muted/40">
          <tr className="text-left">
            <th className="px-3 py-2 font-medium">Region</th>
            <th className="px-3 py-2 font-medium text-right">Dice</th>
            <th className="px-3 py-2 font-medium text-right">IoU</th>
            <th className="px-3 py-2 font-medium text-right">HD95</th>
            <th className="px-3 py-2 font-medium text-right">Vol mm³</th>
          </tr>
        </thead>
        <tbody>
          {REGIONS.map((r) => (
            <tr key={r} className="border-t border-border">
              <td className="px-3 py-2">
                <span
                  className={`mr-2 inline-block h-2 w-2 rounded-sm bg-region-${r}`}
                />
                {REGION_LABELS[r]}
              </td>
              <td className="px-3 py-2 text-right">{fmtNum((m as any)[`dice_${r}`])}</td>
              <td className="px-3 py-2 text-right">{fmtNum((m as any)[`iou_${r}`])}</td>
              <td className="px-3 py-2 text-right">{fmtNum((m as any)[`hd95_${r}`], 2)}</td>
              <td className="px-3 py-2 text-right">{fmtVol((m as any)[`volume_mm3_${r}`])}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
