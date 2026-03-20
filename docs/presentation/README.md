# vmbench Presentation

Standalone reveal.js presentation for vmbench as a formal VM benchmark and inspectable runtime. The narrative and QA state contributions and limitations clearly.

## Open

Open [`../../presentation/index.html`](../../presentation/index.html) in a browser.

If the browser blocks local assets, serve the folder locally:

```bash
python -m http.server 8000 -d presentation
```

Then open [http://localhost:8000](http://localhost:8000).

## Speaker Notes

This deck includes a compact speaker view.

1. Open [`../../presentation/index.html`](../../presentation/index.html)
2. Press `S`
3. A speaker window opens with:
   - current slide
   - next slide
   - large notes text
   - slide counter

## Structure

- [`../../text.md`](../../text.md) - speaker text / narrative
- [`../../QA.md`](../../QA.md) - likely audience questions
- [`../../presentation/index.html`](../../presentation/index.html) - slide deck
- [`../../presentation/speaker.html`](../../presentation/speaker.html) - compact speaker view
- [`../../presentation/css/theme.css`](../../presentation/css/theme.css) - custom deck theme
- [`../../presentation/js/presentation.js`](../../presentation/js/presentation.js) - reveal.js helpers and speaker sync
- [`../../presentation/js/demos.js`](../../presentation/js/demos.js) - runtime demo interactions
- [`../../presentation/assets/demo-runtime-payload.json`](../../presentation/assets/demo-runtime-payload.json) - real runtime payload used by the interactive slide
