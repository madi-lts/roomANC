# roomANC

Whole-room active noise cancellation of highly tonal low-frequency (HVAC)
noise: hardware and software.

The project documentation is a [Quarto](https://quarto.org) website
(cosmo theme, gruvbox colorscheme). Build it with:

```sh
quarto render    # output in _site/
quarto preview   # live local preview
```

Pages:

- `index.qmd` — overview and status
- `physics.qmd` — why global ANC works below the Schroeder frequency, and why passive treatment doesn't
- `spec.qmd` — numbered requirements, architecture, multichannel FxLMS design, verification plan
- `bom.qmd` — bill of materials at three price points (Budget / Mid / Reference)
