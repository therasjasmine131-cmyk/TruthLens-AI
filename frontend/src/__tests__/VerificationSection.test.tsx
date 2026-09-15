import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { VerificationSection } from "../components/analyze/VerificationSection";
import type { Verification } from "../types";

const SAMPLE_VERIFICATION: Verification = {
  status: "completed",
  language: { code: "english", label: "English" },
  claims: [
    {
      text: "The Earth revolves around the Sun.",
      type: "science",
      verdict: "REAL",
      confidence: 0.91,
      confidence_label: "High confidence",
      reason: "Credible sources support the claim.",
      evidence: [
        {
          source_name: "NASA Space Place",
          domain: "spaceplace.nasa.gov",
          title: "What Causes the Seasons?",
          snippet: "Earth orbits the Sun once per year.",
          relation: "SUPPORTS",
          relevance_score: 0.72,
          source_score: 0.9,
          source_tier: "primary",
          url: "https://spaceplace.nasa.gov/seasons/en/",
        },
      ],
      cross_source: {
        distinct_support_domains: 1,
        distinct_contradict_domains: 0,
        independent_support: false,
        independent_contradiction: false,
      },
      ml: { prediction: "REAL", confidence: 0.99 },
    },
    {
      text: "Every college student will receive 50000 rupees every month.",
      type: "education",
      verdict: "FALSE",
      confidence: 0.91,
      confidence_label: "High confidence",
      reason: "Credible sources directly contradict the claim.",
      evidence: [],
      cross_source: {
        distinct_support_domains: 0,
        distinct_contradict_domains: 1,
        independent_support: false,
        independent_contradiction: false,
      },
      ml: { prediction: "FALSE", confidence: 0.98 },
    },
  ],
  overall: {
    verdict: "FALSE",
    confidence: 0.92,
    confidence_label: "High confidence",
    mixed: true,
    counts: { real: 1, false: 1, unverified: 0, total_claims: 2 },
    explanation:
      "The content mixes verified and refuted claims (1 supported, 1 contradicted, 0 unverifiable).",
  },
  evidence_matrix: [
    {
      claim: "The Earth revolves around the Sun.",
      claim_verdict: "REAL",
      evidence_title: "What Causes the Seasons?",
      source: "NASA Space Place",
      domain: "spaceplace.nasa.gov",
      url: "https://spaceplace.nasa.gov/seasons/en/",
      date: null,
      relation: "SUPPORTS",
      relevance: 0.72,
      source_quality: 0.9,
      source_tier: "primary",
      source_reasons: [],
      retrieved_from: "knowledge-base",
    },
  ],
  pipeline: {
    language_detected: "English",
    claims_extracted: 2,
    evidence_items: 1,
    sources_used: ["knowledge-base"],
    live_evidence_used: false,
  },
  ml_article: { prediction: "FALSE", confidence: 0.99, probabilities: { real: 0.01, fake: 0.99 } },
};

function renderVerification() {
  return render(<VerificationSection verification={SAMPLE_VERIFICATION} />);
}

describe("VerificationSection", () => {
  it("shows the overall evidence verdict and confidence", () => {
    renderVerification();
    expect(screen.getAllByText("FALSE").length).toBeGreaterThan(0);
    expect(screen.getByText("92%")).toBeInTheDocument();
    expect(screen.getByText(/mixes verified and refuted claims/i)).toBeInTheDocument();
  });

  it("flags mixed content as a warning", () => {
    renderVerification();
    expect(screen.getByText(/mixed claims detected/i)).toBeInTheDocument();
  });

  it("lists extracted claims with their verdicts", () => {
    renderVerification();
    expect(screen.getByText("Extracted Claims (2)")).toBeInTheDocument();
    expect(screen.getByText("The Earth revolves around the Sun.")).toBeInTheDocument();
    expect(screen.getByText("Every college student will receive 50000 rupees every month.")).toBeInTheDocument();
    // both verdict chips appear
    const chips = screen.getAllByText(/^(REAL|FALSE)$/);
    expect(chips.filter((el) => el.textContent === "REAL").length).toBeGreaterThan(0);
    expect(chips.filter((el) => el.textContent === "FALSE").length).toBeGreaterThan(0);
  });

  it("shows evidence summary counts only (evidence per claim is collapsible)", () => {
    renderVerification();
    expect(screen.getByText(/2 claims extracted/)).toBeInTheDocument();
    expect(screen.getByText(/1 evidence item/)).toBeInTheDocument();
    // claim evidence is collapsed by default
    expect(screen.queryByText(/Supporting evidence \(1\)/i)).not.toBeInTheDocument();
  });

  it("shows the supported/contradicted/unverified counters", () => {
    renderVerification();
    expect(screen.getByText("Supported")).toBeInTheDocument();
    expect(screen.getByText("Contradicted")).toBeInTheDocument();
    expect(screen.getByText("Unverified")).toBeInTheDocument();
    expect(screen.getAllByText("1").length).toBeGreaterThan(0);
  });

  it("expands the evidence matrix table", () => {
    renderVerification();
    const toggle = screen.getByText(/Evidence Matrix \(1 items\)/i);
    // matrix is collapsed by default
    expect(screen.queryByText(/Relevance/)).not.toBeInTheDocument();
    fireEvent.click(toggle);
    expect(screen.getAllByText(/Relevance/i).length).toBeGreaterThan(0);
    expect(screen.getByText("NASA Space Place")).toBeInTheDocument();
    expect(screen.getByText("SUPPORTS")).toBeInTheDocument();
  });

  it("renders only UNVERIFIED styling for a fully-unverified report", () => {
    const unverified: Verification = {
      ...SAMPLE_VERIFICATION,
      claims: [],
      overall: {
        verdict: "UNVERIFIED",
        confidence: 0.4,
        confidence_label: "Evidence insufficient",
        mixed: false,
        counts: { real: 0, false: 0, unverified: 1, total_claims: 1 },
        explanation: "There is not enough credible evidence to confirm or refute the claims.",
      },
      evidence_matrix: [],
    };
    render(<VerificationSection verification={unverified} />);
    expect(screen.getByText("UNVERIFIED")).toBeInTheDocument();
    expect(screen.getByText(/not enough credible evidence/i)).toBeInTheDocument();
    expect(screen.queryByText(/Mixed claims detected/i)).not.toBeInTheDocument();
  });
});