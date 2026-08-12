import { describe, it, expect } from "vitest";
import { formatPercent, confidenceColor, truncate, cn } from "../lib/utils";

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
