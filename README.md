## Development

To start the development server with hot-reload:

```bash
npm run docs:dev
```

The site will be available at `http://localhost:5173`.

## Build

To build the static site for production:

```bash
npm run docs:build
```

The output will be generated in the `.vitepress/dist` directory.

## Preview

To preview the production build locally:

```bash
npm run docs:preview
```

## Deployment

This project is configured to deploy to GitHub Pages via GitHub Actions. See `.github/workflows/deploy.yml` for more details.
