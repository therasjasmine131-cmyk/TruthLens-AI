import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThemeProvider, useTheme } from "../context/ThemeContext";

function Probe() {
  const { resolved, setPreference } = useTheme();
  return (
    <div>
      <span data-testid="resolved">{resolved}</span>
      <button onClick={() => setPreference("dark")}>dark</button>
      <button onClick={() => setPreference("light")}>light</button>
    </div>
  );
}

function Wrapper() {
  return (
    <ThemeProvider>
      <Probe />
    </ThemeProvider>
  );
}

describe("ThemeContext", () => {
  it("toggles theme and persists preference", () => {
    const { getByText, getByTestId } = render(<Wrapper />);
    fireEvent.click(getByText("dark"));
    expect(getByTestId("resolved").textContent).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(localStorage.getItem("truthlens:theme")).toBe("dark");

    fireEvent.click(getByText("light"));
    expect(getByTestId("resolved").textContent).toBe("light");
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(localStorage.getItem("truthlens:theme")).toBe("light");
  });

  it("restores preference from localStorage", () => {
    localStorage.setItem("truthlens:theme", "dark");
    render(<Wrapper />);
    expect(screen.getByTestId("resolved").textContent).toBe("dark");
  });
});
