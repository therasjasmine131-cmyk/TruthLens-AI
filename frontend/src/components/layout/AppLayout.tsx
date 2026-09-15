import { Link, Outlet } from "react-router-dom";
import { Moon, Sun, FlaskConical } from "lucide-react";
import { useTheme } from "../../context/ThemeContext";

export function AppLayout() {
  const { resolved, setPreference } = useTheme();

  return (
    <div className="flex min-h-screen flex-col">
      {/* Slim top bar */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-[#0B1120]/80">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
          <Link to="/" className="flex items-center gap-2">
            <FlaskConical size={18} className="text-primary-600 dark:text-primary-400" />
            <span className="text-base font-bold tracking-tight text-slate-900 dark:text-slate-100">
              TruthLens AI
            </span>
          </Link>
          <button
            onClick={() => setPreference(resolved === "dark" ? "light" : "dark")}
            className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
            aria-label="Toggle theme"
          >
            {resolved === "dark" ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
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
  );
}