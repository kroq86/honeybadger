# vmbench Presentation

Standalone reveal.js presentation for vmbench as an inspectable reasoning runtime. The narrative and QA state contributions and limitations clearly.

## Open

Open [`index.html`](index.html) in a browser.

If the browser blocks local assets, serve the folder locally:

```bash
python -m http.server 8000 -d presentation
```

Then open [http://localhost:8000](http://localhost:8000).

## Speaker Notes

This deck includes a compact speaker view.

1. Open [`index.html`](index.html)
2. Press `S`
3. A speaker window opens with:
   - current slide
   - next slide
   - large notes text
   - slide counter

## Structure

- [`../text.md`](../text.md) - speaker text / narrative
- [`../QA.md`](../QA.md) - likely audience questions
- [`index.html`](index.html) - slide deck
- [`speaker.html`](speaker.html) - compact speaker view
- [`css/theme.css`](css/theme.css) - custom deck theme
- [`js/presentation.js`](js/presentation.js) - reveal.js helpers and speaker sync
- [`js/demos.js`](js/demos.js) - runtime demo interactions
- [`assets/demo-runtime-payload.json`](assets/demo-runtime-payload.json) - real runtime payload used by the interactive slide
