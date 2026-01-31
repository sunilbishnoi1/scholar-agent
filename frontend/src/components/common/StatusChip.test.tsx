// Tests for StatusChip component

import { describe, it, expect } from "vitest";
import { render, screen } from "../../test/utils";
import StatusChip from "./StatusChip";

describe("StatusChip", () => {
  describe("status display", () => {
    it('renders "Ready to Start" for created status', () => {
      render(<StatusChip status="created" />);
      expect(screen.getByText("Ready to Start")).toBeInTheDocument();
    });

    it('renders "Completed" for completed status', () => {
      render(<StatusChip status="completed" />);
      expect(screen.getByText("Completed")).toBeInTheDocument();
    });

    it('renders "Searching..." for searching status', () => {
      render(<StatusChip status="searching" />);
      expect(screen.getByText("Searching...")).toBeInTheDocument();
    });

    it('renders "Analyzing..." for analyzing status', () => {
      render(<StatusChip status="analyzing" />);
      expect(screen.getByText("Analyzing...")).toBeInTheDocument();
    });

    it('renders "Synthesizing..." for synthesizing status', () => {
      render(<StatusChip status="synthesizing" />);
      expect(screen.getByText("Synthesizing...")).toBeInTheDocument();
    });

    it('renders "Error" for error status', () => {
      render(<StatusChip status="error" />);
      expect(screen.getByText("Error")).toBeInTheDocument();
    });

    it('renders "No Papers Found" for error_no_papers_found status', () => {
      render(<StatusChip status="error_no_papers_found" />);
      expect(screen.getByText("No Papers Found")).toBeInTheDocument();
    });
  });

  describe("styling", () => {
    it("applies green styling for completed status", () => {
      render(<StatusChip status="completed" />);
      const chip = screen.getByText("Completed");
      expect(chip).toHaveClass("bg-green-100");
      expect(chip).toHaveClass("text-green-800");
    });

    it("applies red styling for error status", () => {
      render(<StatusChip status="error" />);
      const chip = screen.getByText("Error");
      expect(chip).toHaveClass("bg-red-100");
      expect(chip).toHaveClass("text-red-800");
    });

    it("applies animate-pulse class for in-progress states", () => {
      render(<StatusChip status="searching" />);
      const chip = screen.getByText("Searching...");
      expect(chip).toHaveClass("animate-pulse");
    });

    it("applies rounded-xl border for Ready to Start status", () => {
      render(<StatusChip status="created" />);
      const chip = screen.getByText("Ready to Start");
      expect(chip).toHaveClass("rounded-xl");
      expect(chip).toHaveClass("border");
    });

    it("applies rounded-full for other statuses", () => {
      render(<StatusChip status="completed" />);
      const chip = screen.getByText("Completed");
      expect(chip).toHaveClass("rounded-full");
    });
  });
});
