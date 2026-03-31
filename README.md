# Passport Webcam App

Simple browser app to capture webcam images and export only `JPG/JPEG` under size limits:

- Photograph: maximum `250 KB`
- Signature: maximum `100 KB`

## Run

1. Open `index.html` directly in a modern browser (Chrome/Edge/Firefox/Safari).
2. Click **Start Camera** and allow webcam permission.
3. Choose `Photograph` or `Signature`.
4. Click **Capture**.
5. Click **Download JPG**.

## Notes

- Output is always JPEG (`.jpg`).
- App auto-crops and compresses to fit the selected size limit.
- Photo mode includes an optional AI white-background cleanup.
- If compression fails (rare), retry with cleaner background and better lighting.
