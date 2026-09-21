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
  Bot,
  AlertTriangle,
  Eye,
  EyeOff,
  Sparkles,
} from "lucide-react";
import type { AiStageDecision, ClaimStages, Verdict, Verification, VerificationClaim, VerificationEvidence } from "../../types";
import { Card } from "../ui/Card";
import { cn } from "../../lib/utils";

const AI_DECISION_META: Record<AiStageDecision, { color: string; bg: string }> = {
  SUPPORT: { color: "text-emerald-700 dark:text-emerald-300", bg: "bg-emerald-50 dark:bg-emerald-950/60" },
  CONTRADICT: { color: "text-rose-700 dark:text-rose-300", bg: "bg-rose-50 dark:bg-rose-950/60" },
  INSUFFICIENT: { color: "text-amber-700 dark:text-amber-300", bg: "bg-amber-50 dark:bg-amber-950/60" },
};

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

function AiChip({ label, decision, confidence }: { label: string; decision: AiStageDecision; confidence: number }) {
  const meta = AI_DECISION_META[decision];
  return (
    <span className={cn("inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold", meta.color, meta.bg, "border-current/20")}>
      <Bot size={10} />
      {label}: {decision} · {Math.round(confidence * 100)}%
    </span>
  );
}

const STAGE_KEYS = ["NN_RESULT", "EVIDENCE_RESULT", "AI_RESULT_1", "AI_REVIEW_RESULT", "FINAL_RESULT"] as const;

function StStageRow({ name, value }: { name: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-slate-100 py-1 last:border-0 dark:border-slate-800">
      <span className="font-mono text-[10px] font-semibold text-slate-400 dark:text-slate-500">{name}</span>
      <span className="max-w-[75%] text-right text-[11px] text-slate-700 dark:text-slate-300">{value || "—"}</span>
    </div>
  );
}

function StagesTable({ stages }: { stages: ClaimStages }) {
  const rows: [string, string][] = [
    [
      "NN_RESULT",
      stages.NN_RESULT
        ? `${stages.NN_RESULT.prediction ?? "n/a"} (${stages.NN_RESULT.confidence != null ? Math.round(stages.NN_RESULT.confidence * 100) : "n/a"}%)`
        : "not available",
    ],
    [
      "EVIDENCE_RESULT",
      stages.EVIDENCE_RESULT
        ? `${stages.EVIDENCE_RESULT.verdict} · ${stages.EVIDENCE_RESULT.supporting_count} support / ${stages.EVIDENCE_RESULT.contradicting_count} contra${stages.EVIDENCE_RESULT.independent_sources ? " · independent" : ""}`
        : "no evidence found",
    ],
    [
      "AI_RESULT_1",
      stages.AI_RESULT_1 ? `${stages.AI_RESULT_1.decision} (${Math.round(stages.AI_RESULT_1.confidence * 100)}%)` : "not configured",
    ],
    [
      "AI_REVIEW_RESULT",
      stages.AI_REVIEW_RESULT
        ? `${stages.AI_REVIEW_RESULT.verdict} (${Math.round(stages.AI_REVIEW_RESULT.confidence * 100)}%)${stages.AI_REVIEW_RESULT.agrees_with_first ? "" : " · disagrees with AI#1"}`
        : "not configured",
    ],
    [
      "FINAL_RESULT",
      stages.FINAL_RESULT
        ? `${stages.FINAL_RESULT.verdict} (${Math.round(stages.FINAL_RESULT.confidence * 100)}%) · ${stages.FINAL_RESULT.authority}`
        : "—",
    ],
  ];
  return (
    <div className="mt-3 rounded-md border border-indigo-100 bg-indigo-50/40 px-3 py-2 dark:border-indigo-900/50 dark:bg-indigo-950/20">
      <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-indigo-400 dark:text-indigo-300">
        Developer · stage results
      </p>
      {rows.map(([name, value]) => (
        <StStageRow key={name} name={name} value={value} />
      ))}
    </div>
  );
}

function ClaimRow({ claim, showStages }: { claim: VerificationClaim; showStages: boolean }) {
  const [open, setOpen] = useState(false);
  const evidenceItems = claim.evidence ?? [];
  const ai1 = claim.ai_analysis_1;
  const ai2 = claim.ai_review;
  const hasAi = ai1 != null || ai2 != null;
  const conflicts = claim.conflicts ?? [];
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
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            {claim.type && (
              <span className="inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                {claim.type}
              </span>
            )}
            {claim.final_authority && (
              <span className="inline-block rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-medium text-indigo-600 dark:bg-indigo-950/50 dark:text-indigo-300">
                final: {claim.final_verdict ?? claim.verdict} · {claim.final_authority}
              </span>
            )}
          </div>
        </div>
        <span className="shrink-0 text-slate-400">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </span>
      </button>

      {open && (
        <div className="border-t border-slate-100 px-4 pb-4 pt-3 dark:border-slate-800">
          {evidenceItems.length > 0 && (
            <>
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                Supporting evidence ({evidenceItems.length})
              </p>
              <div className="space-y-2">
                {evidenceItems.map((ev: Record<string, unknown>, idx: number) => (
                  <EvidenceRow key={idx} evidence={ev as unknown as VerificationEvidence} />
                ))}
              </div>
            </>
          )}
          {evidenceItems.length === 0 && (
            <p className="text-xs text-slate-400 dark:text-slate-500">
              No evidence was retrieved for this claim in the current configuration.
            </p>
          )}

          {(hasAi || conflicts.length > 0) && (
            <div className="mt-3 rounded-md border border-slate-100 p-3 dark:border-slate-800">
              <p className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                AI analysis
              </p>
              <div className="flex flex-wrap items-center gap-2">
                {ai1 && <AiChip label="AI#1" decision={ai1.decision} confidence={ai1.confidence} />}
                {ai2 && (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-semibold",
                      ai2.agrees_with_first
                        ? "text-emerald-700 dark:text-emerald-300"
                        : "text-rose-700 dark:text-rose-300",
                      ai2.agrees_with_first
                        ? "bg-emerald-50 dark:bg-emerald-950/60"
                        : "bg-rose-50 dark:bg-rose-950/60",
                    )}
                  >
                    <Bot size={10} />
                    AI#2 {ai2.verdict} · {Math.round(ai2.confidence * 100)}%
                    {ai2.agrees_with_first ? " · agrees" : " · disagrees"}
                  </span>
                )}
                {ai1?.reasoning && (
                  <span className="text-[11px] italic text-slate-500 dark:text-slate-400">{ai1.reasoning}</span>
                )}
              </div>
              {ai2 && ai2.problems.length > 0 && (
                <p className="mt-2 text-[11px] text-amber-700 dark:text-amber-300">
                  Reviewer notes: {ai2.problems.join("; ")}
                </p>
              )}
              {conflicts.length > 0 && (
                <ul className="mt-2 space-y-1">
                  {conflicts.map((conflict, idx) => (
                    <li
                      key={idx}
                      className="flex items-start gap-1.5 text-[11px] text-amber-700 dark:text-amber-300"
                    >
                      <AlertTriangle size={11} className="mt-0.5 shrink-0" />
                      {conflict}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {showStages && claim.stages && <StagesTable stages={claim.stages} />}
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
  const [showStages, setShowStages] = useState(false);
  const overall = verification.overall;
  const claims = verification.claims ?? [];
  const matrix = verification.evidence_matrix ?? [];
  const pipeline = verification.pipeline;
  const pipelineStages = verification.stages?.PIPELINE ?? null;

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
              {pipeline.ai_used && (
                <span className="inline-flex items-center gap-1 rounded bg-emerald-50 px-1.5 py-0.5 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
                  <Bot size={11} />
                  AI analysis · {pipeline.ai_claims_analyzed ?? 0} claim{pipeline.ai_claims_analyzed !== 1 ? "s" : ""}
                </span>
              )}
              {pipeline.ai_used === false && (
                <span className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">
                  <Bot size={11} />
                  AI stages not configured
                </span>
              )}
              <button
                onClick={() => setShowStages((v) => !v)}
                className={cn(
                  "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-medium transition-colors",
                  showStages
                    ? "border-indigo-300 bg-indigo-50 text-indigo-600 dark:border-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300"
                    : "border-slate-200 text-slate-500 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800",
                )}
              >
                {showStages ? <EyeOff size={11} /> : <Eye size={11} />}
                Developer: stages
              </button>
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
        {showStages && pipelineStages && (
          <div className="border-t border-slate-200 px-4 py-3 dark:border-slate-800">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
              Pipeline stages
            </p>
            <div className="mt-1 flex flex-wrap items-center gap-1 text-[10px] font-mono text-slate-500 dark:text-slate-400">
              {pipelineStages.map((stage, idx) => (
                <span key={stage} className="inline-flex items-center gap-1">
                  {idx > 0 && <span className="text-slate-300 dark:text-slate-600">→</span>}
                  <span
                    className={cn(
                      "rounded px-1.5 py-0.5",
                      idx === pipelineStages.length - 1
                        ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                        : "bg-slate-100 dark:bg-slate-800",
                    )}
                  >
                    {stage}
                  </span>
                </span>
              ))}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-5">
              {STAGE_KEYS.map((name) => {
                const count = claims.filter((c) => c.stages && c.stages[name] != null).length;
                return (
                  <div key={name} className="rounded border border-slate-100 px-2 py-1 text-center dark:border-slate-800">
                    <span className="block text-[9px] font-mono font-semibold uppercase text-slate-400">{name}</span>
                    <span className="block text-[11px] font-semibold text-slate-700 dark:text-slate-200">
                      {count}/{claims.length}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {/* AI final validation */}
      {verification.gemini_validation && verification.gemini_validation.label && (
        <Card className="overflow-hidden border-indigo-200 dark:border-indigo-900/60">
          <div className="flex items-center gap-2 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
            <Sparkles size={16} className="text-indigo-500" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                AI Final Validation
              </h3>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                After the pipeline result, the AI independently reviewed the article and explains why.
              </p>
            </div>
          </div>
          <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-start">
            <div className="flex flex-col items-start gap-2 sm:min-w-[210px]">
              <span className="inline-flex items-center gap-1.5 rounded-md border border-indigo-300 bg-indigo-50 px-3 py-1.5 text-xs font-bold text-indigo-700 dark:border-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300">
                <Bot size={14} />
                Validates {verification.gemini_validation.label === "REAL" ? "REAL" : verification.gemini_validation.label === "FALSE" ? "FAKE" : "UNVERIFIED"}
              </span>
              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                AI confidence{" "}
                {verification.gemini_validation.confidence_score != null
                  ? `${verification.gemini_validation.confidence_score}%`
                  : `${Math.round(verification.gemini_validation.confidence * 100)}%`}
              </p>
              {verification.gemini_validation.initial_model_verdict &&
                verification.gemini_validation.initial_model_verdict !== "n/a" && (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold",
                      verification.gemini_validation.initial_model_was_correct
                        ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                        : "bg-rose-50 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300",
                    )}
                  >
                    Initial model said {verification.gemini_validation.initial_model_verdict === "FALSE" ? "FAKE" : verification.gemini_validation.initial_model_verdict} ·{" "}
                    {verification.gemini_validation.initial_model_was_correct
                      ? "confirmed"
                      : "overridden"}
                  </span>
                )}
              {verification.gemini_validation.agrees ? (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
                  <CheckCircle2 size={11} />
                  Agrees with the evidence verdict
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700 dark:bg-amber-950/50 dark:text-amber-300">
                  <AlertTriangle size={11} />
                  Disagrees — reviewed independently
                </span>
              )}
            </div>
            <div className="flex-1">
              <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
                {verification.gemini_validation.reasoning || "No reasoning returned."}
              </p>
              {verification.gemini_validation.key_claims_verified &&
                verification.gemini_validation.key_claims_verified.length > 0 && (
                  <div className="mt-3">
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                      Key claims verified
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {verification.gemini_validation.key_claims_verified.slice(0, 4).map((kc, idx) => (
                        <span
                          key={idx}
                          className="flex items-start gap-1.5 rounded border border-slate-100 bg-slate-50/70 px-2 py-1 text-[11px] text-slate-600 dark:border-slate-800 dark:bg-slate-900/40 dark:text-slate-300"
                        >
                          <span
                            className={cn(
                              "mt-px shrink-0 rounded px-1 py-px text-[9px] font-bold",
                              kc.status === "SUPPORTED"
                                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                                : kc.status === "CONTRADICTED"
                                  ? "bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300"
                                  : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300",
                            )}
                          >
                            {kc.status}
                          </span>
                          <span className="min-w-0 flex-1">{kc.claim}</span>
                          {kc.evidence_strength && kc.evidence_strength !== "—" && (
                            <span className="shrink-0 text-[9px] font-semibold text-slate-400">
                              {kc.evidence_strength}
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              {verification.gemini_validation.sources_checked &&
                verification.gemini_validation.sources_checked.length > 0 && (
                  <div className="mt-3">
                    <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
                      Sources checked ({verification.gemini_validation.sources_checked.length})
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {verification.gemini_validation.sources_checked.slice(0, 6).map((src, idx) => (
                        <span
                          key={idx}
                          className="flex items-start gap-1.5 rounded border border-slate-100 bg-slate-50/70 px-2 py-1 text-[11px] text-slate-600 dark:border-slate-800 dark:bg-slate-900/40 dark:text-slate-300"
                        >
                          <span
                            className={cn(
                              "mt-px shrink-0 rounded px-1 py-px text-[9px] font-bold",
                              src.supports_claim
                                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300"
                                : "bg-rose-100 text-rose-700 dark:bg-rose-950/50 dark:text-rose-300",
                            )}
                          >
                            {src.supports_claim ? "SUPPORTS" : "CONTRADICTS"}
                          </span>
                          {src.url ? (
                            <a
                              href={src.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex min-w-0 flex-1 items-center gap-0.5 text-primary-600 hover:underline dark:text-primary-400"
                            >
                              <span className="truncate">{src.title}</span>
                              <ExternalLink size={10} className="shrink-0" />
                            </a>
                          ) : (
                            <span className="min-w-0 flex-1">{src.title || "Source"}</span>
                          )}
                          {src.source_type && (
                            <span className="shrink-0 text-[9px] uppercase text-slate-400">
                              {src.source_type}
                            </span>
                          )}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          </div>
        </Card>
      )}

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
              <ClaimRow key={idx} claim={claim} showStages={showStages} />
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