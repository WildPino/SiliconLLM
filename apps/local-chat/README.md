# SiliconLLM Local Chat

SiliconLLM Local Chat is a native C++17 and Qt Widgets desktop shell for one locally approved inference runtime. It has no browser view or network inference. The catalog starts empty on purpose: the current research checkout does not yet contain an admitted CPU runtime for this app. The deterministic Python process used by the tests is a protocol fixture, not a usable model.

The runtime contract and qualification receipt are documented in [`docs/local-chat/RUNTIME_PROTOCOL.md`](../../docs/local-chat/RUNTIME_PROTOCOL.md). The product goal and acceptance gates remain in [`docs/local-chat/GOAL.md`](../../docs/local-chat/GOAL.md).

## Build on Windows

Use CMake 3.21 or newer, Ninja, Python 3, Qt 6.8.3 for MinGW 64-bit, and the matching Qt MinGW 13.1 toolchain (`mingw1310_64`). The Qt and compiler versions must match; a different MinGW installation can produce incompatible Qt binaries. The checked build configuration used:

- Qt: `%LOCALAPPDATA%\SiliconLLM\devtools\Qt\6.8.3\mingw_64`
- Compiler tools: `%LOCALAPPDATA%\SiliconLLM\devtools\Qt\Tools\mingw1310_64\bin`
- CMake generator: Ninja

From the repository root in PowerShell, set the local paths and add the compiler and Qt runtime directories to `PATH`:

```powershell
$qtPath = Join-Path $env:LOCALAPPDATA 'SiliconLLM\devtools\Qt\6.8.3\mingw_64'
$compilerPath = Join-Path $env:LOCALAPPDATA 'SiliconLLM\devtools\Qt\Tools\mingw1310_64\bin'
$env:Path = "$compilerPath;$qtPath\bin;$env:Path"
```

The build wrapper configures CMake, builds the app and runs its tests by default:

```powershell
.\apps\local-chat\build.ps1 `
  -QtPath $qtPath `
  -CompilerPath $compilerPath `
  -BuildDirectory '.\apps\local-chat\build'
```

The wrapper runs tests unless `-SkipTests` is specified. It needs Qt's Core, Gui, Widgets, Concurrent and Test components, plus Python 3 for the deterministic engine fixture. To assemble a deployable application folder, including Qt's runtime dependencies:

```powershell
.\apps\local-chat\package.ps1 `
  -QtPath $qtPath `
  -CompilerPath $compilerPath `
  -BuildDirectory '.\apps\local-chat\build' `
  -Destination '.\dist\local-chat'
```

The package wrapper deploys an already-built executable and does not run tests. The result is a Windows application folder, not a signed installer. The compiler `bin` and Qt `bin` directories are added to `PATH` by the wrapper. CMake and Ninja must be available on `PATH` for the build wrapper.

The equivalent core CMake setup, useful when diagnosing a wrapper failure, is:

```powershell
$env:Path = "$compilerPath;$qtPath\bin;$env:Path"
cmake -S apps/local-chat -B apps/local-chat/build -G Ninja `
  -DCMAKE_PREFIX_PATH="$qtPath" `
  -DCMAKE_CXX_COMPILER="$compilerPath\g++.exe" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build apps/local-chat/build --config Release
ctest --test-dir apps/local-chat/build -C Release --output-on-failure
```

If Qt, compiler, generator or build type changes, configure into a fresh build directory. CMake caches absolute toolchain paths in `CMakeCache.txt`.

## Start and import a catalog

After a successful build, start `apps/local-chat/build/silicon-chat.exe`. The application accepts:

```text
silicon-chat.exe --catalog path\to\models.json --data-dir path\to\chat-data
```

`--catalog` opens the supplied local JSON catalog. Without it, the app uses the last catalog selected through **Models**; before one is saved, the list is empty. `--data-dir` overrides the conversation directory for this run. Use `--help` to see all options. `--capture path.png` and `--capture-settings path.png` save verification screenshots and exit. `--theme light|dark` selects and saves the appearance.

## Appearance and controls

**Settings** uses segmented buttons for light/dark themes and text size, plus a generation statistics toggle. Preferences are saved locally. Both themes share one fixed visual style. Glass-like separation uses opaque surfaces and static borders: no blur, shadow effects, decorative animation or idle redraw timer. The first streaming text appears immediately; later changes are grouped every 75 ms without postponing updates when deltas arrive faster. Markdown is formatted on completion.

The model selector, **Models** and **Settings** sit above the conversation. Send switches to Stop during generation. Right-click a conversation (or use the keyboard context-menu key) to rename or delete it. The current **Models** action imports a local catalog; downloading models is a future feature described in [`DISTRIBUTION.md`](../../docs/local-chat/DISTRIBUTION.md).

The optional `LOCAL_CHAT_BUILD_UI_PROBE=ON` CMake target and `tests/local_chat/benchmark-ui.ps1` compare native window CPU and memory with a synthetic transcript. They are development tools, excluded from the portable package. Results and limits are in [`VALIDATION.md`](../../docs/local-chat/VALIDATION.md).

The checked-in [`resources/catalog.example.json`](resources/catalog.example.json) contains `{"schemaVersion":1,"models":[]}`. It is a valid empty catalog and does not contain fake model paths or approval hashes. To add a runtime, place a catalog beside its local executable, model and tokenizer files, then select it in the app or pass it to `--catalog`. Relative paths in the file resolve from the catalog's directory. Each model must use the `silicon-chat-v1` protocol and include an admitted CPU-runtime receipt bound to the exact runtime executable, artifact, tokenizer and generation settings. Optional helper, adapter, template and tokenizer configuration files also need matching SHA-256 entries. See the protocol document for the complete schema and validation rules. The app does not download, repair or silently substitute model files.

## Conversation data

By default, Qt stores application data in the current user's local application data area, under a `local-chat\conversations` directory. Each conversation is a versioned JSON file named by its UUID. Saves use an atomic replacement, so a write interruption does not leave a partially written transcript. User, assistant and system messages, interruption state, and the runtime statistics are kept. Invalid files are reported and left for recovery; they are not silently deleted. Use `--data-dir` to select a different location or make a backup.

## Troubleshooting

- **The model list is empty.** An empty list is expected until an admitted CPU runtime has a valid catalog entry. The example catalog is deliberately empty.
- **A model is unavailable.** Check the catalog and receipt paths, executable, artifact, tokenizer and working directory. Hashes and runtime settings must match the receipt. Catalog verification runs outside the window thread and hashes files in chunks; validating a large model can take time.
- **The model never becomes ready.** The process must write only protocol JSON events to standard output. Send diagnostic logs to standard error. Its first event must declare the matching `silicon-chat-v1` protocol, context limit and streaming capability.
- **A prompt is too large.** The runtime adapter must tokenize the complete transcript with the receipt's tokenizer and template, then check prompt plus requested output against the context before inference. It must return a clear error rather than truncate or summarize. The current process bridge does not perform tokenization or this context calculation itself.
- **No text appears until generation ends.** A non-streaming model returns only a completed response. A streaming model must send real delta events during generation; the app does not synthesize them.
- **Stop or closing the app interrupts generation.** On Windows, the owned process job is terminated and the next request starts a fresh runtime process. The adapter should not create children before readiness or outside the owned job.
- **Conversation data is missing.** Check the directory passed to `--data-dir` and its write permissions. The default location is provided by Qt's `AppLocalDataLocation`; it can vary by Windows account and application settings.

The current UI, process and storage tests exercise a deterministic fixture. They do not establish real model quality, tokenizer parity with a research reference, cancellation behavior of a production inference engine, or the proposed five-percent CPU overhead gate. Those require a separately admitted runtime and measurements on the target machine.
