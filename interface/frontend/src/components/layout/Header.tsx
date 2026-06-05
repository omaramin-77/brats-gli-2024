import { Link } from "@tanstack/react-router";
import { Github, Activity } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import { Button } from "@/components/ui/button";
import { useHealth } from "@/api/queries";
import { Badge } from "@/components/ui/badge";

export function Header() {
  const { data: health } = useHealth();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between px-6">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-primary/15 text-primary">
            <Activity className="h-4 w-4" />
          </span>
          <span>BraTS GLI-2024</span>
          <span className="text-muted-foreground font-normal text-sm hidden md:inline">
            Tumour Segmentation
          </span>
        </Link>

        <nav className="hidden md:flex items-center gap-6 text-sm">
          <Link to="/dashboard" className="text-muted-foreground hover:text-foreground" activeProps={{ className: "text-foreground font-medium" }}>
            Dashboard
          </Link>
          <Link to="/about" className="text-muted-foreground hover:text-foreground" activeProps={{ className: "text-foreground font-medium" }}>
            About
          </Link>
        </nav>

        <div className="flex items-center gap-2">
          {health && (
            <Badge variant="outline" className="hidden md:inline-flex font-mono text-[10px]">
              <span className={`mr-1.5 h-1.5 w-1.5 rounded-full ${health.cuda_available ? "bg-success" : "bg-warning"}`} />
              {health.device.toUpperCase()} · {health.models_enabled}/{health.models_total}
            </Badge>
          )}
          <a href="https://github.com/niivue/niivue" target="_blank" rel="noreferrer">
            <Button variant="ghost" size="icon" aria-label="GitHub">
              <Github className="h-4 w-4" />
            </Button>
          </a>
          <ThemeToggle />
          <Link to="/dashboard">
            <Button size="sm" className="hidden sm:inline-flex">Start Analysis</Button>
          </Link>
        </div>
      </div>
    </header>
  );
}
