import { describe, it, expect } from "vitest";
import { formatPercent, confidenceColor, truncate, cn, computeTextStats } from "../lib/utils";

describe("formatPercent", () => {
  it("formats fractions as percentages", () => {
    expect(formatPercent(0.924)).toBe("92.4%");
    expect(formatPercent(1)).toBe("100.0%");
    expect(formatPercent(0)).toBe("0.0%");
  });
});

describe("confidenceColor", () => {
  it("returns appropriate classes for each band", () => {
    expect(confidenceColor(0.92)).toContain("emerald");
    expect(confidenceColor(0.6)).toContain("amber");
    expect(confidenceColor(0.3)).toContain("slate");
  });
});

describe("truncate", () => {
  it("truncates long strings", () => {
    expect(truncate("hello world", 5)).toBe("hell…");
  });
  it("keeps short strings", () => {
    expect(truncate("hi")).toBe("hi");
  });
});

describe("cn", () => {
  it("merges classes and drops falsy values", () => {
    expect(cn("a", false && "b", "c")).toBe("a c");
  });
});

describe("computeTextStats", () => {
  const HEADLINE = "Tamil Nadu inks MoUs worth Rs 67,000 crore";
  it("counts empty input as all zeros", () => {
    expect(computeTextStats("", "")).toEqual({
      chars: 0,
      words: 0,
      sentences: 0,
      unique_words: 0,
      vocabulary_richness: 0,
      average_sentence_length: 0,
      capitalized_words: 0,
      exclamation_marks: 0,
      question_marks: 0,
    });
  });
  it("counts headline-only text (matches backend text_stats)", () => {
    const s = computeTextStats(HEADLINE, "");
    expect(s.chars).toBe(42);
    expect(s.words).toBe(8);
    expect(s.sentences).toBe(1);
    expect(s.unique_words).toBe(8);
    expect(s.vocabulary_richness).toBe(1);
  });
  it("counts combined headline + article consistently", () => {
    const article = "The deals were announced. Officials expect new jobs.";
    const s = computeTextStats(HEADLINE, article);
    // 8 headline tokens + 8 article tokens = 16 words.
    expect(s.words).toBe(16);
    expect(s.chars).toBe(article.length + 1 + HEADLINE.length);
    expect(s.sentences).toBe(2);
  });
});
