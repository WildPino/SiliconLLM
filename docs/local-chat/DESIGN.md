# Local chat visual design

The 0.2.1 interface is entirely in English, including status messages, runtime errors, accessibility labels and command-line help. It uses native Qt Widgets with opaque surfaces and static luminous edges. Light and dark themes share one visual style; there is no border-style preference. Existing border preferences are ignored and removed when settings are saved.

## Hierarchy and controls

Conversation content is the main canvas. Navigation and controls occupy a separate compact layer, following the hierarchy described in [Apple, Meet Liquid Glass](https://developer.apple.com/videos/play/wwdc2025/219/). This application uses a static visual interpretation, without blur or live transparency.

The sidebar contains New chat and conversation history. Rename and Delete live in the conversation context menu. The model selector uses a flat painted field, a small chevron and a spaced list. Settings uses exclusive segmented buttons for Light/Dark and Small/Medium/Large, plus generation statistics visibility. Send changes to Stop during generation.

The conversation uses a system font, a constrained reading width and message heights that follow their content. Copy actions support messages and code. Theme and text changes apply when settings are accepted; appearance controls are disabled during generation.

## Resource budget

No blur, graphics shadows, decorative animation, idle redraw loop, browser view or downloaded fonts. The model field paints only when Qt requests it. Segmented settings reuse standard buttons, keyboard focus and button-group exclusivity. The streaming timer and runtime scheduling remain unchanged in this revision.

[VALIDATION.md](VALIDATION.md) contains test evidence and the measured 0.2.0 UI comparison. Those measurements do not qualify real CPU model decode overhead or establish a new numeric comparison for 0.2.1.
