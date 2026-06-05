import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  compareAll,
  compareModels,
  createSession,
  deleteSession,
  getHealth,
  getModels,
  getPredictionMetrics,
  getSession,
  runPrediction,
  uploadFile,
} from "./client";
import type { UploadKind } from "@/lib/types";

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
    retry: 1,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    staleTime: 60 * 60 * 1000,
  });
}

export function useSessionInfo(sid: string | null) {
  return useQuery({
    queryKey: ["session", sid],
    queryFn: () => getSession(sid!),
    enabled: !!sid,
  });
}

export function useCreateSession() {
  return useMutation({
    mutationFn: createSession,
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteSession() {
  return useMutation({
    mutationFn: (sid: string) => deleteSession(sid),
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useUpload(sid: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (args: { kind: UploadKind; file: File; onProgress?: (p: number) => void }) => {
      if (!sid) throw new Error("No session");
      return uploadFile(sid, args.kind, args.file, args.onProgress);
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session", sid] });
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRunPrediction(sid: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (modelKey: string) => {
      if (!sid) throw new Error("No session");
      return runPrediction(sid, modelKey);
    },
    onSuccess: (data) => {
      qc.setQueryData(["prediction", sid, data.model_key], data);
      qc.invalidateQueries({ queryKey: ["session", sid] });
      toast.success(`Inference complete: ${data.model_short_name}`);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function usePrediction(sid: string | null, modelKey: string | null) {
  return useQuery({
    queryKey: ["prediction", sid, modelKey],
    queryFn: () => getPredictionMetrics(sid!, modelKey!),
    enabled: !!sid && !!modelKey,
    staleTime: Infinity,
  });
}

export function useCompare(sid: string | null) {
  return useMutation({
    mutationFn: async (keys: string[]) => {
      if (!sid) throw new Error("No session");
      return compareModels(sid, keys);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useCompareAll(sid: string | null) {
  return useMutation({
    mutationFn: async () => {
      if (!sid) throw new Error("No session");
      return compareAll(sid);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
