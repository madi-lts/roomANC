# roomANC — project context

Whole-room active noise cancellation for a low-frequency HVAC drone in a
rental. The project documentation is a Quarto website in this repo — read
`physics.qmd` (modal regime, why ANC is viable below ~150 Hz, why passive
treatment mostly is not), `spec.qmd` (requirements and the multichannel
FxLMS design), `measurements.qmd`, `bom.qmd`, and `layout.qmd` for current
state.

## Working conventions

- The user is a senior physicist (trapped-ion QC background): don't simplify
  the math, but define every symbol before first use.
- Rental constraints: no structural modifications; nothing attached to the
  building fabric beyond caulk/weatherstripping.
- Candidate workstreams:
  1. Measurement/analysis script (spectrum, tonal peak detection, level vs. position).
  2. Rectangular-room modal model (mode frequencies, pressure distribution, predicted nulls).
  3. FxLMS prototype (single- then multichannel, with secondary-path identification).
  4. Tuned absorber (Helmholtz/membrane) calculator as a passive fallback.
