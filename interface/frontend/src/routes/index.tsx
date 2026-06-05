import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Brain, Layers, GitMerge } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useModels } from "@/api/queries";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "BraTS GLI-2024 — Brain MRI Tumour Segmentation" },
      { name: "description", content: "Research dashboard for deep-learning segmentation of brain tumours on multimodal MRI." },
    ],
  }),
  component: Landing,
});

function MriGlyph() {
  return (
    <div className="relative h-72 w-72 sm:h-96 sm:w-96">
      <motion.div
        className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/20 via-accent/10 to-transparent blur-2xl"
        animate={{ scale: [1, 1.05, 1], opacity: [0.6, 0.9, 0.6] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      />
      <svg viewBox="0 0 320 320" className="relative h-full w-full">
        <defs>
          <radialGradient id="brain" cx="50%" cy="45%">
            <stop offset="0%" stopColor="hsl(217 91% 60%)" stopOpacity="0.9" />
            <stop offset="65%" stopColor="hsl(217 91% 40%)" stopOpacity="0.6" />
            <stop offset="100%" stopColor="hsl(217 91% 20%)" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="160" cy="160" r="130" fill="url(#brain)" />
        {[0, 1, 2, 3].map((i) => (
          <motion.ellipse
            key={i}
            cx="160"
            cy="160"
            rx={110 - i * 22}
            ry={70 - i * 14}
            fill="none"
            stroke="hsl(190 90% 60%)"
            strokeOpacity={0.5 - i * 0.1}
            strokeWidth="1"
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 24 + i * 8, repeat: Infinity, ease: "linear" }}
            style={{ transformOrigin: "160px 160px" }}
          />
        ))}
        <motion.circle
          cx="160"
          cy="160"
          r="14"
          fill="#dc323c"
          animate={{ scale: [1, 1.4, 1], opacity: [0.85, 1, 0.85] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
          cx="180"
          cy="150"
          r="22"
          fill="#ffa500"
          fillOpacity={0.45}
          animate={{ scale: [1, 1.15, 1] }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.circle
          cx="160"
          cy="160"
          r="38"
          fill="#32b4dc"
          fillOpacity={0.2}
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
        />
      </svg>
    </div>
  );
}

function Landing() {
  const { data } = useModels();
  const families = [
    {
      icon: Brain,
      title: "Modality experts",
      desc: "Four single-modality networks — one each for T1n, T1c, T2w, and T2-FLAIR.",
      count: data?.models.filter((m) => m.badge === "Modality expert").length ?? 4,
    },
    {
      icon: Layers,
      title: "Naive baseline",
      desc: "Reference multi-modal model trained on all four MRI sequences concatenated.",
      count: data?.models.filter((m) => m.badge === "Baseline").length ?? 1,
    },
    {
      icon: GitMerge,
      title: "Late fusion",
      desc: "Ensembles the four modality experts at the logit level. Coming soon.",
      count: data?.models.filter((m) => m.badge === "Coming soon").length ?? 1,
      soon: true,
    },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        <section className="mx-auto max-w-[1400px] px-6 pt-12 pb-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <Badge variant="outline" className="mb-4 font-mono text-[10px]">
                BraTS GLI-2024 · Research prototype
              </Badge>
              <h1 className="text-4xl sm:text-5xl font-bold tracking-tight leading-tight">
                Brain MRI Tumour Segmentation
              </h1>
              <p className="mt-5 text-base text-muted-foreground max-w-xl leading-relaxed">
                A research dashboard for running deep-learning segmentation models on 3D NIfTI MRI
                volumes. Upload four standard sequences, pick a model, and compare predicted
                whole-tumour, tumour-core, and enhancing-tumour masks against ground truth — with
                Dice, IoU, and HD95 computed server-side.
              </p>
              <div className="mt-7 flex flex-wrap gap-3">
                <Link to="/dashboard">
                  <Button size="lg" className="gap-2">
                    Start analysis <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
                <Link to="/about">
                  <Button size="lg" variant="outline">How it works</Button>
                </Link>
              </div>
            </motion.div>
            <div className="flex justify-center lg:justify-end">
              <MriGlyph />
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1400px] px-6 pb-24">
          <h2 className="text-xs font-semibold tracking-widest uppercase text-muted-foreground mb-4">
            Model families
          </h2>
          <div className="grid md:grid-cols-3 gap-4">
            {families.map((f) => (
              <Card key={f.title} className="p-5">
                <div className="flex items-start justify-between mb-3">
                  <div className="grid h-9 w-9 place-items-center rounded-md bg-primary/15 text-primary">
                    <f.icon className="h-4 w-4" />
                  </div>
                  {f.soon ? (
                    <Badge variant="outline">Coming soon</Badge>
                  ) : (
                    <Badge variant="secondary" className="font-mono">{f.count}</Badge>
                  )}
                </div>
                <h3 className="font-semibold">{f.title}</h3>
                <p className="mt-1 text-sm text-muted-foreground leading-relaxed">{f.desc}</p>
              </Card>
            ))}
          </div>
        </section>
      </main>
      <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground">
        Research prototype — not for clinical use. Trained on BraTS GLI-2024.
      </footer>
    </div>
  );
}
