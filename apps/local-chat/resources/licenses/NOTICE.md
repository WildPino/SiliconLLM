# Dependency notices

The portable Windows package uses dynamically linked Qt 6.8.3 Core, Gui and
Widgets libraries and their required platform, style and image plugins.
Qt sources for this version are available from
[Qt qtbase v6.8.3](https://github.com/qt/qtbase/tree/v6.8.3).
The accompanying LGPL 3 and GPL 3 license texts are included here.

The matching MinGW 13.1 compiler package supplies libgcc, libstdc++ and
winpthreads runtime libraries. GCC sources and the runtime exception are at
[GCC 13.1](https://github.com/gcc-mirror/gcc/tree/releases/gcc-13.1.0).
The winpthreads copyright and permission notices are included in
winpthreads-COPYING.txt; sources are in
[MinGW winpthreads](https://github.com/mingw-w64/mingw-w64/tree/master/mingw-w64-libraries/winpthreads).

The application source is distributed under the repository LICENSE.
No model weights, tokenizer packages, Python runtime or research artifacts
are bundled in the desktop shell.
