# Handoff: Low-frequency HVAC drone mitigation (apartment)

## Problem

Apartment in Greenbelt, MD has a constant low-frequency drone from rooftop HVAC units and pumps. Goal: reduce the perceived level in a bedroom/living space, as a tenant (no structural modifications). Landlord escalation is a parallel track; a measured spectrum supports both.

## Physics conclusions so far

Notation: f = frequency (Hz), c ≈ 343 m/s speed of sound, λ = c/f wavelength, V = room volume (m³), T₆₀ = reverberation time (s), L = a room dimension (m), m = partition surface mass (kg/m²).

- **Regime.** Below the Schroeder frequency f_S ≈ 2000·√(T₆₀/V) (≈180 Hz for V ≈ 50 m³, T₆₀ ≈ 0.4 s), the room field is modal, not diffuse. In the modal regime, controlling a few mode amplitudes gives global (whole-room) cancellation. Above it, ANC only creates a ~λ/10 quiet zone around each error mic — not viable for a room.
- **Active noise control (ANC) is feasible** if the drone is a few stable tones below ~150 Hz. Expected 10–20 dB reduction on the tones across most of the room; weak on broadband content. Car-cabin engine-order cancellation is the commercial precedent.
- **Passive absorption is ineffective** at these frequencies: porous absorbers need particle velocity, which is zero at a wall and maximal at λ/4 (≈1.4 m at 60 Hz). Thin panels/foam do nothing. Only deep corner traps (60–100 cm rockwool) or **tuned resonant absorbers** (Helmholtz or membrane, tuned to the drone tone; narrow bandwidth, defeated by pitch drift) help.
- **Passive blocking is weak** at low f: mass law TL ≈ 20·log₁₀(f·m) − 47 dB. Leaks (window frame, outlets, penetrations) dominate; acoustic caulk / mass-loaded vinyl over the window / door weatherstripping are the cheap airborne fixes.
- **Structure-borne path** (vibration through slab/framing, radiated by room surfaces) is bypassed by all in-room passive treatment. Diagnostic: open a window; if the drone barely changes, it is mostly structure-borne and the fix is source isolation (landlord). Damping modes (lower Q) does make ANC easier to stabilize — passive and active are complementary.
- **Modal geometry.** Hard walls are pressure antinodes (normal velocity = 0 ⇒ pressure maximum). For the lowest axial mode along L, p(x) ∝ cos(πx/L): loudest at the walls, null at L/2. Corners are the loudest points in the room. Bed placement near a wall makes it worse, not better. Quiet zones for the actual drone frequency must be found empirically.

## Proposed ANC design

- **Measure first:** spectrum at pillow height with a measurement mic (or phone app as a first pass). Identify tone frequencies, stability/drift, and broadband floor. Also map level vs. position in the room.
- **Actuators:** 2–4 subwoofers in room corners (couple to all modes).
- **Error sensors:** 4–8 microphones distributed around the room.
- **Reference signal:** accelerometer on the wall/window facing the units, or synthesized reference from tracked tone frequency (adaptive notch / phase-locked oscillator) — tonal noise is predictable, so causality is not a constraint.
- **Controller:** multichannel filtered-x LMS (FxLMS) on a low-latency platform (Bela, Teensy, dedicated DSP, or Linux with a real-time audio stack). Requires offline secondary-path identification (speaker→mic transfer functions) with periodic re-identification as the room changes. Must track HVAC pitch drift adaptively.

## Suggested next tasks (pick as needed)

1. Measurement/analysis script: record audio, compute averaged spectrum, detect tonal peaks, log level vs. position.
2. Room-mode simulation: rectangular room modal model given L×W×H, predict mode frequencies and pressure distribution, overlay measured peaks, find predicted nulls.
3. FxLMS prototype: single-channel then multichannel, with secondary-path identification routine, simulated first against the modal model, then on hardware.
4. Tuned absorber calculator: Helmholtz / membrane absorber dimensions for a target frequency as a passive fallback.

## Constraints

- Rental: no structural changes, nothing attached to the building fabric beyond caulk/weatherstripping.
- User is a senior physicist (trapped-ion QC background, strong theory); no need to simplify the math. Define every symbol before first use.
