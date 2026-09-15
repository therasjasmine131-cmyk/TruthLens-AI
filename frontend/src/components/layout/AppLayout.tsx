import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Menu, Moon, Sun, FlaskConical } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { useTheme } from "../../context/ThemeContext";

export function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { resolved, setPreference } = useTheme();

  return (
    <div className="flex min-h-screen">
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />

      <div className="flex min-w-0 flex-1 flex-col lg:pl-60">
        {/* Mobile top bar */}
        <header className="sticky top-0 z-40 flex items-center justify-between border-b border-slate-200 bg-white/80 px-4 py-3 backdrop-blur dark:border-slate-800 dark:bg-[#0B1120]/80 lg:hidden">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMobileOpen(true)}
              className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
              aria-label="Open menu"
            >
              <Menu size={20} />
            </button>
            <div className="flex items-center gap-1.5">
              <FlaskConical size={16} className="text-primary-600 dark:text-primary-400" />
              <span className="text-sm font-bold text-slate-900 dark:text-slate-100">TruthLens AI</span>
            </div>
          </div>
          <button
            onClick={() => setPreference(resolved === "dark" ? "light" : "dark")}
            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
            aria-label="Toggle theme"
          >
            {resolved === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </header>

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
          <Outlet />
        </main>

        <footer className="mx-auto w-full max-w-7xl px-4 pb-6 text-center text-[11px] text-slate-400 dark:text-slate-500 sm:px-6 lg:px-8">
          This is an ML-based prediction, not proof of factual truth. TruthLens AI reflects
          statistical patterns learned from a training dataset and can be wrong. Verify important
          claims using reliable sources.
        </footer>
      </div>
    </div>
  );
}
