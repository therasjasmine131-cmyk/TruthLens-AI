import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { ResultPanel } from "../components/analyze/ResultPanel";
import type { AnalysisResult } from "../types";

const SAMPLE_RESULT: AnalysisResult = {
  prediction: "REAL",
  confidence: 0.924,
  confidence_level: "Very High Confidence",
  probabilities: { real: 0.924, fake: 0.051, uncertain: 0.025 },
  model_raw: { p_real: 0.948, p_fake: 0.052 },
  model: "Random Forest",
  model_info: { name: "Random Forest", n_features: 5000, metrics: { accuracy: 0.997 } },
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
    method: "coefficients",
    model_class: "LogisticRegression",
    features: [{ term: "reuters", weight: 1.2, contribution: 0.14, influence: "positive" }],
    note: "Note",
    direction_label: "Positive influence pushes toward REAL.",
  },
  disclaimer: "Educational disclaimer",
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
  it("renders prediction, confidence and probability cards", () => {
    renderPanel();
    expect(screen.getAllByText("REAL").length).toBeGreaterThan(0);
    expect(screen.getAllByText("92.4%").length).toBeGreaterThan(0);
    expect(screen.getByText("Very High Confidence")).toBeInTheDocument();
    expect(screen.getByText("Probability Distribution")).toBeInTheDocument();
  });

  it("renders article statistics and TF-IDF keywords", () => {
    renderPanel();
    expect(screen.getByText("Word Count")).toBeInTheDocument();
    expect(screen.getAllByText("reuters").length).toBeGreaterThan(0);
    expect(screen.getAllByText("election").length).toBeGreaterThan(0);
  });

  it("shows the model name and metrics", () => {
    renderPanel();
    expect(screen.getByText("Random Forest")).toBeInTheDocument();
    expect(screen.getByText("99.7%")).toBeInTheDocument();
  });
});
