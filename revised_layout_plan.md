Now I have the full picture. I'll rebuild `index.html` with a proper two-column layout — symptom checker on the left, article sidebar on the right — and remove the broken section at the bottom.Done. The page is now a clean two-column layout:

```
┌─────────────────────────────────────────┬──────────────────────┐
│  NAV: ◆ WellnessCheck   Checker   Blog  │                      │
├─────────────────────────────────────────┤                      │
│                                         │  📰 Wellness Blog    │
│  ◆ WellnessCheck                        │  View all →          │
│  Symptom Checker                        │  ──────────────────  │
│  Choose a dataset below...              │  [All][Prev][Cond]   │
│                                         │  [Symp][Derm][Tech]  │
│  [📊 Kaggle]  [🧬 DDXPlus]             │  ──────────────────  │
│                                         │  🫀 Prevention       │
│  ┌──────────────────────────────────┐   │  Inside the AHA      │
│  │ Describe your symptoms           │   │  PREVENT Calc…      │
│  │ [textarea]                       │   │  7 min · Jul 8       │
│  │ [Analyse symptoms]               │   │                      │
│  └──────────────────────────────────┘   │  ❤️ Prevention      │
│                                         │  Have You Checked    │
│  ┌──────────────┐ ┌──────────────┐      │  Your PREVENT…      │
│  │ Top          │ │ All models   │      │                      │
│  │ predictions  │ │ comparison   │      │  🩸 Prevention       │
│  └──────────────┘ └──────────────┘      │  Type 2 Diabetes…   │
│                                         │                      │
│  ┌──────────────────────────────────┐   │  🤖 Technology       │
│  │ Feedback                         │   │  How ML Is…         │
│  └──────────────────────────────────┘   │  ...                 │
├─────────────────────────────────────────┴──────────────────────┤
│  FOOTER                                                        │
└────────────────────────────────────────────────────────────────┘
```

**Sidebar features:**
- Sticky — stays visible as the user scrolls through results
- Category filter pills — All / Prevention / Conditions / Symptoms / Dermatology / Technology
- Articles sorted newest first, each showing emoji, category (colour-coded), title, and read time
- Clicking any article opens it in the full article reader
- Hides automatically on mobile (≤900px) so it doesn't crowd the checker