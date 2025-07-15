# Changelog

## 2025-07-14 — "Virtual Vibes & Dependency Drip"

### What We Did
- 🕵️‍♂️ Diagnosed and resolved Python environment chaos (conda, Homebrew, system Python, .venv all fighting for attention)
- 🚫 Abandoned conda in favor of a clean `.venv` setup for maximum compatibility
- 🛠️ Installed all dependencies (including `streamlit-image-select`) in `.venv`
- 🧹 Cleaned up and attempted to remove the conda environment (permission issues noted)
- 🏄‍♂️ Successfully launched the Streamlit app natively on Mac, confirmed MPS (Apple Silicon) acceleration
- 🎉 Lip sync app is now working as intended!

### TODO
- [ ] Remove any lingering conda or system Python references from scripts or docs
- [ ] Test on a fresh Mac to confirm reproducibility
- [ ] Consider adding a script to automate `.venv` setup for new users
- [ ] Monitor for any further dependency drift

### Reflections
Today was a journey through the wilds of Python packaging. From conda confusion to `.venv` zen, we surfed the dependency waves and came out on top. The app is now running smooth, and the docs are fresher than a pour-over at a Brooklyn café. Onward to more AI lip-sync magic! 