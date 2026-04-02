# Passport Webcam App

Simple browser app to capture webcam images and export only `JPG/JPEG` under size limits:

- Photograph: `630 x 810` pixels, maximum `250 KB`
- Signature: maximum `100 KB`

## Run

1. Open `index.html` directly in a modern browser (Chrome/Edge/Firefox/Safari).
2. Click **Start Camera** and allow webcam permission.
3. Choose `Photograph` or `Signature`.
4. Either click **Capture** (webcam) or **Upload Image** (existing photo).
5. Click **Download JPG**.

## Mobile Rear Camera Page

- Open `mobile.html` for a mobile-optimized flow.
- Tap **Start Back Camera** to request the rear (`environment`) camera.
- Use the on-screen face outline to align and then capture.
- Output is JPG at `630 x 810` under `250 KB`.

## Notes

- Output is always JPEG (`.jpg`).
- App auto-crops and compresses to fit the selected size limit.
- Photo mode includes an optional AI white-background cleanup.
- Photo mode now includes ICAO assist for head-size framing and frontal-pose guidance.
- Mobile page is intended for HTTPS/localhost contexts to ensure camera access.
- If compression fails (rare), retry with cleaner background and better lighting.
