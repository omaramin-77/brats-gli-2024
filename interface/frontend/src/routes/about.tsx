import { createFileRoute } from "@tanstack/react-router";
import { Header } from "@/components/layout/Header";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useModels } from "@/api/queries";
import { MODALITY_LABELS } from "@/lib/types";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — BraTS GLI-2024" },
      { name: "description", content: "Background on modalities, sub-regions, and models used in this project." },
    ],
  }),
  component: AboutPage,
});

function AboutPage() {
  const { data: models } = useModels();
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 mx-auto max-w-3xl w-full px-6 py-10">
        <h1 className="text-3xl font-bold tracking-tight">About this project</h1>
        <p className="mt-3 text-muted-foreground leading-relaxed">
          A graduation-defense research dashboard for brain-tumour segmentation on the BraTS
          GLI-2024 dataset. Inference runs server-side; this UI lets you upload volumes, pick a
          model, and inspect predicted masks against ground truth.
        </p>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-3">Imaging modalities</h2>
          <div className="grid sm:grid-cols-2 gap-3">
            {(["t1n", "t1c", "t2w", "t2f"] as const).map((k) => (
              <Card key={k} className="p-4">
                <div className="font-mono text-xs uppercase text-primary">{k}</div>
                <div className="font-semibold mt-1">{MODALITY_LABELS[k].name}</div>
                <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                  {MODALITY_LABELS[k].desc}. Used as one channel of the 3D input volume passed to
                  the segmentation network.
                </p>
              </Card>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-3">Tumour sub-regions</h2>
          <div className="space-y-2">
            {[
              { k: "wt", label: "Whole Tumour (WT)", desc: "All tumour tissue — oedema plus the tumour core." },
              { k: "tc", label: "Tumour Core (TC)", desc: "Non-enhancing and enhancing tumour, excluding oedema." },
              { k: "et", label: "Enhancing Tumour (ET)", desc: "Contrast-enhancing portion seen on T1c." },
            ].map((r) => (
              <Card key={r.k} className="p-4 flex items-start gap-3">
                <span className={`h-3 w-3 rounded-sm mt-1.5 bg-region-${r.k}`} />
                <div>
                  <div className="font-semibold">{r.label}</div>
                  <p className="text-sm text-muted-foreground leading-relaxed">{r.desc}</p>
                </div>
              </Card>
            ))}
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-lg font-semibold mb-3">Model architectures</h2>
          <ul className="space-y-2">
            {models?.models.map((m) => (
              <li key={m.key}>
                <Card className="p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="font-semibold">{m.display_name}</div>
                      <div className="font-mono text-xs text-muted-foreground">{m.architecture}</div>
                    </div>
                    <Badge variant="outline">{m.badge}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">{m.description}</p>
                </Card>
              </li>
            ))}
          </ul>
        </section>

        <p className="mt-12 text-xs text-muted-foreground text-center">
          Research prototype — not for clinical use. Trained on BraTS GLI-2024.
        </p>
      </main>
    </div>
  );
}
