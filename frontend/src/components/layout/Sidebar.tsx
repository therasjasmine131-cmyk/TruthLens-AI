import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  ScanSearch,
  History,
  BarChart3,
  Gauge,
  Database,
  GraduationCap,
  Settings,
  FlaskConical,
  GitCompareArrows,
  ScanText,
  Newspaper,
  X,
} from "lucide-react";
import { cn } from "../../lib/utils";
import { useApi } from "../../hooks/useApi";
import { api } from "../../api/client";
import type { HealthStatus } from "../../types";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/analyze", label: "Analyze News", icon: ScanSearch },
  { to: "/trending", label: "Trending News", icon: Newspaper },
  { to: "/history", label: "Prediction History", icon: History },
  { to: "/batch", label: "Batch Analysis", icon: ScanText },
  { to: "/compare", label: "Compare Analyses", icon: GitCompareArrows },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/model-performance", label: "Model Performance", icon: Gauge },
  { to: "/dataset", label: "Dataset Explorer", icon: Database },
  { to: "/methodology", label: "Methodology", icon: GraduationCap },
  { to: "/settings", label: "Settings", icon: Settings },
];

function StatusDot({ healthy, label, sub }: { healthy: boolean; label: string; sub: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={cn(
          "h-2 w-2 rounded-full",
          healthy ? "bg-emerald-500" : "bg-rose-500",
          healthy && "shadow-[0_0_6px_rgba(16,185,129,0.6)]",
        )}
      />
      <div className="leading-tight">
        <p className="text-[11px] font-medium text-slate-300">{label}</p>
        <p className="text-[10px] text-slate-500">{sub}</p>
      </div>
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { data: health } = useApi<HealthStatus>(() => api.health(), []);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary-600 text-white shadow-sm">
          <FlaskConical size={18} />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-bold tracking-tight text-white">TruthLens AI</p>
          <p className="text-[10px] text-slate-400">AI-powered news credibility analysis</p>
        </div>
      </div>

      <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                isActive
                  ? "bg-primary-600/15 text-primary-300"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200",
              )
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="border-t border-slate-800/80 px-5 py-4">
        <div className="flex flex-col gap-3">
          <StatusDot
            healthy={health?.model?.status === "online"}
            label="AI Model"
            sub={health?.model?.status === "online" ? "Online" : health?.model?.status === "error" ? "Error" : "Offline"}
          />
          <StatusDot
            healthy={health?.database?.healthy === true}
            label="Database"
            sub={health?.database?.healthy ? "Connected" : "Disconnected"}
          />
        </div>
      </div>
    </div>
  );
}

export function Sidebar({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  return (
    <>
      {/* Desktop */}
      <aside className="hidden h-screen w-60 shrink-0 bg-[#0B1120] lg:block">
        <SidebarContent />
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm" onClick={onClose} />
          <aside className="absolute inset-y-0 left-0 w-64 bg-[#0B1120] shadow-xl animate-slide-in">
            <button
              onClick={onClose}
              className="absolute right-3 top-4 rounded p-1.5 text-slate-400 hover:text-white"
              aria-label="Close menu"
            >
              <X size={18} />
            </button>
            <SidebarContent onNavigate={onClose} />
          </aside>
        </div>
      )}
    </>
  );
}
