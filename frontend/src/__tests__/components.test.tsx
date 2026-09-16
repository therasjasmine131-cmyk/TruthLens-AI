import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProgressSteps } from "../components/analyze/ProgressSteps";
import { EmptyState } from "../components/ui/EmptyState";
import { PredictionBadge } from "../components/ui/PredictionBadge";

const STEPS = ["Cleaning text", "Encoding tokens", "Running neural network", "Calculating confidence"];

describe("ProgressSteps", () => {
  it("marks completed and active steps", () => {
    render(<ProgressSteps steps={STEPS} activeStep={1} />);
    expect(screen.getByText("Cleaning text")).toBeInTheDocument();
    expect(screen.getByText("Running neural network")).toBeInTheDocument();
    expect(screen.getByText("✓")).toBeInTheDocument();
  });
});

describe("EmptyState", () => {
  it("renders title and description", () => {
    render(<EmptyState title="No analyses yet" description="Analyze your first article." />);
    expect(screen.getByText("No analyses yet")).toBeInTheDocument();
    expect(screen.getByText("Analyze your first article.")).toBeInTheDocument();
  });
});

describe("PredictionBadge", () => {
  it("renders the label for all classes", () => {
    for (const label of ["REAL", "FAKE", "UNCERTAIN"] as const) {
      const { unmount } = render(<PredictionBadge label={label} />);
      expect(screen.getByText(label)).toBeInTheDocument();
      unmount();
    }
  });
});
