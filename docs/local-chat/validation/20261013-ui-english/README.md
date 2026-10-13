# English interface and controls — 0.2.1

Four CTest suites passed: store 8, engine 18, UI 15, controller 6 QtTest results
(including suite initialization and cleanup). Logs are saved in this directory.

The settings test selects the exclusive theme/text buttons, applies them,
checks persistence and verifies that no dropdown or border toggle remains in
the settings dialog. A legacy border preference is ignored and removed on save.
All application status/error strings, labels, accessibility labels and CLI help
are English. User conversation content and imported model metadata are preserved.

The packaged executable was launched four times with only Windows directories
in PATH: light/dark window and light/dark settings screenshots, all exit code 0.
Screenshots were visually inspected. The model selector uses a static painted
field and standard Qt list interaction; settings use native checkable buttons.
There are no new timers, animations, graphics effects or background work.

The idle probe JSON files are single 1000ms synthetic UI samples, not a repeated
performance comparison or real model overhead qualification. The 0.2.0 benchmark
remains historical evidence for that version. Runtime logic and streaming cadence
are unchanged; engine/controller changes in this revision translate strings only.
