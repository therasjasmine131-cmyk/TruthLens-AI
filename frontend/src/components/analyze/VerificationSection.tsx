import { useState } from "react";
import {
  CheckCircle2,
  XCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  ShieldAlert,
  Lightbulb,
  Layers,
  FileSearch,
} from "lucide-react";
import type { Verdict, Verification, VerificationClaim, VerificationEvidence } from "../../types";
import { Card } from "../ui/Card";
import { cn } from "../../lib/utils";

const VERDICT_META: Record<Verdict, { label: string; color: string; bg: string; border: string; icon: typeof CheckCircle2 }> = {
  REAL: {
    label: "REAL",
    color: "text-emerald-700 dark:text-emerald-300",
    bg: "bg-emerald-50 dark:bg-emerald-950/60",
    border: "border-emerald-300 dark:border-emerald-700",
    icon: CheckCircle2,
  },
  FALSE: {
    label: "FALSE",
    color: "text-rose-700 dark:text-rose-300",
    bg: "bg-rose-50 dark:bg-rose-950/60",
    border: "border-rose-300 dark:border-rose-700",
    icon: XCircle,
  },
  UNVERIFIED: {
    label: "UNVERIFIED",
    color: "text-amber-700 dark:text-amber-300",
    bg: "bg-amber-50 dark:bg-amber-950/60",
    border: "border-amber-300 dark:border-amber-700",
    icon: HelpCircle,
  },
};

function VerdictBadge({ verdict, size = "lg" }: { verdict: Verdict; size?: "sm" | "lg" }) {
  const meta = VERDICT_META[verdict];
  const Icon = meta.icon;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border font-bold",
        meta.color,
        meta.bg,
        meta.border,
        size === "lg" ? "px-4 py-1.5 text-base" : "px-2 py-0.5 text-xs",
      )}
    >
      <Icon size={size === "lg" ? 18 : 13} />
      {meta.label}
    </span>
  );
}

function ConfidenceBar({ confidence, verdict }: { confidence: number; verdict: Verdict }) {
  return (
    <div className="w-full max-w-xs">
      <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
        <span>Confidence</span>
        <span className="tabular-nums font-semibold text-slate-700 dark:text-slate-200">
          {Math.round(confidence * 100)}%
        </span>
      </div>
      <div className="mt-1 h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-500",
            verdict === "REAL" ? "bg-emerald-500" : verdict === "FALSE" ? "bg-rose-500" : "bg-amber-500",
          )}
          style={{ width: `${Math.round(confidence * 100)}%` }}
        />
      </div>
    </div>
  );
}

function ClaimRow({ claim }: { claim: VerificationClaim }) {
  const [open, setOpen] = useState(false);
  const evidenceItems = claim.evidence ?? [];
  return (
    <div className="rounded-lg border border-slate-200 bg-white dark:border-slate-800 dark:bg-[#111a2e]">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 p-4 text-left"
      >
        <VerdictBadge verdict={claim.verdict} size="sm" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium leading-snug text-slate-800 dark:text-slate-100">
            {claim.text}
          </p>
          <p className="mt-1 line-clamp-2 text-[11px] text-slate-500 dark:text-slate-400">
            {claim.reason}
          </p>
          {claim.type && (
            <span className="mt-1 inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">
              {claim.type}
            </span>
          )}
        </div>
        <span className="shrink-0 text-slate-400">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {open && evidenceItems.length > 0 && (
        <div className="border-t border-slate-100 px-4 pb-4 pt-3 dark:border-slate-800">
          <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
            Supporting evidence ({evidenceItems.length})
          </p>
          <div className="space-y-2">
            {evidenceItems.map((ev: Record<string, unknown>, idx: number) => (
              <EvidenceRow key={idx} evidence={ev as unknown as VerificationEvidence} />
            ))}
          </div>
        </div>
      )}
      {open && evidenceItems.length === 0 && (
        <div className="border-t border-slate-100 px-4 pb-4 pt-3 dark:border-slate-800">
          <p className="text-xs text-slate-400 dark:text-slate-500">
            No evidence was retrieved for this claim in the current configuration.
          </p>
        </div>
      )}
    </div>
  );
}

function EvidenceRow({ evidence }: { evidence: VerificationEvidence }) {
  const relationColor =
    evidence.relation === "SUPPORTS"
      ? "text-emerald-600 dark:text-emerald-400"
      : evidence.relation === "CONTRADICTS"
        ? "text-rose-600 dark:text-rose-400"
        : "text-slate-500 dark:text-slate-400";
  const relationBg =
    evidence.relation === "SUPPORTS"
      ? "bg-emerald-50 dark:bg-emerald-950/40"
      : evidence.relation === "CONTRADICTS"
        ? "bg-rose-50 dark:bg-rose-950/40"
        : "bg-slate-50 dark:bg-slate-900/60";
  return (
    <div className="flex items-start gap-2 rounded-md border border-slate-100 p-2.5 text-xs dark:border-slate-800">
      <span className={`shrink-0 rounded px-1.5 py-0.5 font-semibold ${relationBg} ${relationColor}`}>
        {evidence.relation}
      </span>
      <div className="min-w-0 flex-1">
        <p className="font-medium text-slate-700 dark:text-slate-200">
          {evidence.source ?? evidence.domain ?? "Unknown source"}
        </p>
        <p className="mt-0.5 line-clamp-2 text-[11px] text-slate-500 dark:text-slate-400">
          {evidence.evidence_title}
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-slate-400 dark:text-slate-500">
          <span>Relevance: {Math.round(evidence.relevance * 100)}%</span>
          <span>Source: {(evidence.source_quality * 100).toFixed(0)}%</span>
          <span className={`font-medium ${
            evidence.source_tier === "primary"
              ? "text-emerald-600 dark:text-emerald-400"
              : evidence.source_tier === "secondary"
                ? "text-amber-600 dark:text-amber-400"
                : "text-slate-500"
          }`}>
            {evidence.source_tier}
          </span>
          {evidence.url && (
            <a
              href={evidence.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-0.5 text-primary-600 hover:underline dark:text-primary-400"
            >
              <ExternalLink size={10} />
              source
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

export function VerificationSection({ verification }: { verification: Verification }) {
  const [showMatrix, setShowMatrix] = useState(false);
  const overall = verification.overall;
  const claims = verification.claims ?? [];
  const matrix = verification.evidence_matrix ?? [];
  const pipeline = verification.pipeline;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Overall verdict */}
      <Card className="overflow-hidden">
        <div className="flex flex-col items-center gap-4 p-6 text-center sm:flex-row sm:text-left">
          <div className="flex flex-col items-center gap-2 sm:items-start">
            <VerdictBadge verdict={overall.verdict} />
            <ConfidenceBar confidence={overall.confidence} verdict={overall.verdict} />
          </div>
          <div className="flex-1">
            <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
              {overall.explanation}
            </p>
            {overall.mixed && (
              <div className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-medium text-amber-700 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-300">
                <ShieldAlert size={13} />
                Mixed claims detected — some supported, some contradicted.
              </div>
            )}
            <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-500 dark:text-slate-400">
              <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">
                <FileSearch size={11} />
                {pipeline.claims_extracted} claim{pipeline.claims_extracted !== 1 ? "s" : ""} extracted
              </span>
              <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">
                <Layers size={11} />
                {pipeline.evidence_items} evidence item{pipeline.evidence_items !== 1 ? "s" : ""}
              </span>
              <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">
                <Lightbulb size={11} />
                ML used as secondary signal
              </span>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 divide-x border-t border-slate-200 text-center text-xs dark:border-slate-800">
          <div className="px-3 py-2">
            <span className="block text-lg font-bold text-emerald-600 dark:text-emerald-400">
              {overall.counts.real}
            </span>
            <span className="text-[10px] text-slate-400">Supported</span>
          </div>
          <div className="px-3 py-2">
            <span className="block text-lg font-bold text-rose-600 dark:text-rose-400">
              {overall.counts.false}
            </span>
            <span className="text-[10px] text-slate-400">Contradicted</span>
          </div>
          <div className="px-3 py-2">
            <span className="block text-lg font-bold text-amber-600 dark:text-amber-400">
              {overall.counts.unverified}
            </span>
            <span className="text-[10px] text-slate-400">Unverified</span>
          </div>
        </div>
      </Card>

      {/* Claims */}
      {claims.length > 0 && (
        <Card>
          <div className="border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              Extracted Claims ({claims.length})
            </h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Atomic claims identified from the submitted text.
            </p>
          </div>
          <div className="space-y-3 p-5">
            {claims.map((claim, idx) => (
              <ClaimRow key={idx} claim={claim} />
            ))}
          </div>
        </Card>
      )}

      {/* Evidence matrix */}
      {matrix.length > 0 && (
        <Card>
          <button
            onClick={() => setShowMatrix((v) => !v)}
            className="flex w-full items-center justify-between border-b border-slate-200 px-5 py-4 text-left dark:border-slate-800"
          >
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                Evidence Matrix ({matrix.length} items)
              </h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                Sources retrieved and evaluated against each claim.
              </p>
            </div>
            {showMatrix ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showMatrix && (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-100 text-[10px] uppercase tracking-wide text-slate-400 dark:border-slate-800 dark:text-slate-500">
                    <th className="px-4 py-2">Claim</th>
                    <th className="px-4 py-2">Source</th>
                    <th className="px-4 py-2">Relation</th>
                    <th className="px-4 py-2">Relevance</th>
                    <th className="px-4 py-2">Source quality</th>
                  </tr>
                </thead>
                <tbody>
                  {matrix.map((ev, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-slate-50 last:border-0 dark:border-slate-800/60"
                    >
                      <td className="max-w-[200px] truncate px-4 py-2 text-slate-700 dark:text-slate-300">
                        {ev.claim}
                      </td>
                      <td className="px-4 py-2 text-slate-600 dark:text-slate-300">
                        {ev.source ?? ev.domain ?? "—"}
                      </td>
                      <td className="px-4 py-2">
                        <span
                          className={cn(
                            "rounded px-1.5 py-0.5 font-medium",
                            ev.relation === "SUPPORTS"
                              ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400"
                              : ev.relation === "CONTRADICTS"
                                ? "bg-rose-50 text-rose-700 dark:bg-rose-950/40 dark:text-rose-400"
                                : "bg-slate-50 text-slate-600 dark:bg-slate-800 dark:text-slate-400",
                          )}
                        >
                          {ev.relation}
                        </span>
                      </td>
                      <td className="tabular-nums px-4 py-2 text-slate-600 dark:text-slate-300">
                        {Math.round(ev.relevance * 100)}%
                      </td>
                      <td className="px-4 py-2 text-slate-600 dark:text-slate-300">
                        {ev.source_tier}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}