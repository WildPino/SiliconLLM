# SiliconLLM desktop preview

Open `silicon-chat.exe` to start the native desktop window. This folder includes
the required Qt and compiler runtime libraries; no separate Qt installation is
needed to open the application. Keep the executable and its accompanying files
together.

This preview implements the chat window, local conversation history, runtime
process management and statistics. It does not bundle a model. No CPU chat
runtime is currently admitted for this preview, so its initial model list is
empty and generation remains unavailable until a qualified runtime is installed.
There is no simulated model or simulated performance display.

Use **Models** to select the catalog of an admitted runtime.
Catalogs bind the executable, weights, tokenizer and runtime settings to a CPU
qualification receipt. The included `catalog.example.json` is deliberately
empty. The adapter contract is in
[RUNTIME_PROTOCOL.md](docs/local-chat/RUNTIME_PROTOCOL.md).

Open **Settings** for light/dark themes, text size, statistics. These preferences are saved locally. Right-click a conversation
to rename or delete it. Send becomes Stop during generation. Model downloads
are planned and are not included in this preview.

The application saves conversations in the current user's local application
data directory. To choose a directory and catalog explicitly, run:

```text
silicon-chat.exe --catalog models.json --data-dir chat-data
```

Stop interrupts the owned runtime process group. Its next request reloads the
model. Already received partial text is kept and marked as interrupted.

`LICENSE.txt` contains the application license. The `dependency-licenses`
folder contains the notices for the bundled runtime libraries. Application
source and build instructions are in the
[SiliconLLM repository](https://github.com/WildPino/SiliconLLM).
