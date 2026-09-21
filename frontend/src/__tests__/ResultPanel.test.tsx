import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ResultPanel } from "../components/analyze/ResultPanel";
import type { AnalysisResult } from "../types";

const SAMPLE_RESULT: AnalysisResult = {
  prediction: "REAL",
  confidence: 0.924,
  confidence_level: "Very High Confidence",
  decided: true,
  uncertain_threshold: 0.78,
  probabilities: { real: 0.924, fake: 0.076, uncertain: 0.0 },
  model_raw: { p_real: 0.924, p_fake: 0.076 },
  model: "Neural Network (BiGRU)",
  model_info: {
    name: "Neural Network (BiGRU)",
    n_features: 20000,
    metrics: { accuracy: 0.994 },
  },
  keywords: [
    { term: "reuters", score: 0.12 },
    { term: "election", score: 0.09 },
  ],
  article_stats: {
    word_count: 120,
    character_count: 700,
    sentence_count: 8,
    average_sentence_length: 15,
    unique_words: 80,
    vocabulary_richness: 0.67,
    capitalized_words: 5,
    exclamation_marks: 0,
    question_marks: 1,
  },
  explanation: {
    method: "nn-token-influence",
    model_class: "BiGRU (Embedding->BiGRU->Dense)",
    features: [{ term: "reuters", weight: 1.2, contribution: 0.14, influence: "positive" }],
    note: "Model-associated features note.",
    direction_label: "These are model-associated features.",
  },
  disclaimer: "This is an AI model prediction, not proof of factual truth.",
  saved: true,
  history_id: 1,
};

function renderPanel() {
  return render(
    <MemoryRouter>
      <ResultPanel result={SAMPLE_RESULT} compact />
    </MemoryRouter>
  );
}

describe("ResultPanel", () => {
  it("renders the single end-to-end flow map", () => {
    renderPanel();
    expect(screen.getByText("How TruthLens Works")).toBeInTheDocument();
    expect(screen.getByText("One end-to-end pipeline — every step is executed and logged on the backend.")).toBeInTheDocument();
    expect(screen.getByText(/Claim extraction — split the text into atomic checkable claims/)).toBeInTheDocument();
    expect(screen.getByText(/Gemini decides TRUE or FALSE using live data and its own knowledge/)).toBeInTheDocument();
    expect(screen.getByText(/overall verdict — Gemini's final answer is shown/i)).toBeInTheDocument();
  });

  it("does not show a manufactured UNCERTAIN percentage for a decided prediction", () => {
    renderPanel();
    // prediction is REAL, so the UNCERTAIN abstain card must NOT be shown.
    expect(screen.queryByText(/UNCERTAIN \(abstain\)/i)).not.toBeInTheDocument();
    // The distribution is honest: it never claims UNCERTAIN is a percentage.
    expect(screen.queryByText(/residual margin/i)).not.toBeInTheDocument();
  });

  it("renders article statistics and neural keywords", () => {
    renderPanel();
    expect(screen.getByText("Word Count")).toBeInTheDocument();
    expect(screen.getAllByText("reuters").length).toBeGreaterThan(0);
    expect(screen.getAllByText("election").length).toBeGreaterThan(0);
  });

  it("shows the model name and honest test-set metrics", () => {
    renderPanel();
    expect(screen.getByText("Neural Network (BiGRU)")).toBeInTheDocument();
    expect(screen.getByText("99.4%")).toBeInTheDocument();
    expect(screen.getByText("Model Performance on Test Dataset")).toBeInTheDocument();
  });

  it("renders a final verdict hero when evidence verdict is present", () => {
    render(
      <MemoryRouter>
        <ResultPanel
          result={{ ...SAMPLE_RESULT, verdict: "FALSE" }}
          compact
        />
      </MemoryRouter>
    );
    expect(screen.getByText("Final Verdict")).toBeInTheDocument();
    expect(screen.getByText("FALSE")).toBeInTheDocument();
  });

  it("does not show a manufactured abstain card for an undecided result", () => {
    render(
      <MemoryRouter>
        <ResultPanel
          result={{ ...SAMPLE_RESULT, prediction: "UNCERTAIN", decided: false, confidence: 0.6 }}
          compact
        />
      </MemoryRouter>
    );
    expect(screen.queryByText(/UNCERTAIN \(abstain\)/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/abstain decision/i)).not.toBeInTheDocument();
  });
});
