# AVTrack Project Page

Static site for the [AVTrack: Audio-Visual Tracking in Human-centric Complex Scenes](https://github.com/FudanCVL/AVTrack) project (ICML 2026).

This branch (`gh-pages`) is served by GitHub Pages at
<https://fudancvl.github.io/AVTrack/>.

## Editing

Everything lives in this branch only — `main` has the code, `gh-pages` has the website.

```
index.html              # All page content
static/
├── images/             # Figures (teaser, source, challenge, ...) + favicon + hero GIF
├── videos/             # Reserved for future video assets
├── css/                # Bulma + project styles
└── js/                 # FontAwesome bundle + minor scripts
```

Local preview:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Built on

[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) by Eliahu Horwitz, adapted from the [Nerfies](https://nerfies.github.io) project page.
