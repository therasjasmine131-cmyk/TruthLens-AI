import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { CheckCircle2, AlertCircle, X } from "lucide-react";
import { cn } from "../lib/utils";

type ToastKind = "success" | "error" | "info";

interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  description?: string;
}

interface ToastContextValue {
  toast: (kind: ToastKind, title: string, description?: string) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);
let nextId = 1;

const KIND_STYLES: Record<ToastKind, { icon: typeof CheckCircle2; ring: string; text: string }> = {
  success: { icon: CheckCircle2, ring: "border-emerald-500/30", text: "text-emerald-500" },
  error: { icon: AlertCircle, ring: "border-rose-500/30", text: "text-rose-500" },
  info: { icon: CheckCircle2, ring: "border-primary-500/30", text: "text-primary-500" },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    (kind: ToastKind, title: string, description?: string) => {
      const id = nextId++;
      setToasts((prev) => [...prev, { id, kind, title, description }]);
      window.setTimeout(() => dismiss(id), 4500);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ toast, dismiss }), [toast, dismiss]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-full max-w-sm flex-col gap-2">
        {toasts.map((t) => {
          const style = KIND_STYLES[t.kind];
          const Icon = style.icon;
          return (
            <div
              key={t.id}
              role="status"
              className={cn(
                "pointer-events-auto flex items-start gap-3 rounded-lg border bg-white p-3.5 shadow-card-hover dark:bg-[#111a2e]",
                style.ring,
                "animate-slide-in",
              )}
            >
              <Icon className={cn("mt-0.5 h-5 w-5 shrink-0", style.text)} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">{t.title}</p>
                {t.description && (
                  <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{t.description}</p>
                )}
              </div>
              <button
                onClick={() => dismiss(t.id)}
                className="rounded p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                aria-label="Dismiss notification"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}
