#include "mainwindow.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <QSettings>
#include <QTemporaryDir>
#include <QTimer>
#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

using namespace silicon;

struct ProcessCost { double cpuMs = 0; double privateBytes = 0; double workingSetBytes = 0; };
ProcessCost processCost() {
    ProcessCost result;
#ifdef Q_OS_WIN
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
        const auto ticks = [](FILETIME time) {
            return (static_cast<quint64>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
        };
        result.cpuMs = (ticks(kernel) + ticks(user)) / 10000.0;
    }
    PROCESS_MEMORY_COUNTERS_EX memory{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) {
        result.privateBytes = static_cast<double>(memory.PrivateUsage);
        result.workingSetBytes = static_cast<double>(memory.WorkingSetSize);
    }
#endif
    return result;
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("SiliconLLM-UI-Test");
    QCoreApplication::setApplicationName("Native-rendering-probe");
    QTemporaryDir preferences;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, preferences.path());
    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addOption({"scenario", "idle or stream (synthetic UI only)", "name", "idle"});
    parser.addOption({"theme", "light or dark", "name", "light"});
    parser.addOption({"output", "Measurement JSON file", "file"});
    parser.addOption({"capture", "Optional synthetic fixture screenshot", "png"});
    parser.addOption({"duration", "Measurement duration in milliseconds", "ms", "3000"});
    parser.process(app);
    const auto scenario = parser.value("scenario");
    const auto theme = parser.value("theme");
    bool validDuration = false;
    const auto duration = parser.value("duration").toInt(&validDuration);
    if ((scenario != "idle" && scenario != "stream") || (theme != "light" && theme != "dark") ||
        !validDuration || duration < 1000 || duration > 30000 || !parser.isSet("output")) return 2;
    MainWindow window;
    window.resize(1180, 780);
#ifdef UI_COST_HAS_THEME
    window.setTheme(theme);
#else
    if (theme != "light") return 2;
#endif
    ModelSpec spec;
    spec.id = "render-fixture"; spec.name = "Fixture di rendering — nessuna inferenza";
    spec.approved = true; spec.available = true; spec.maxContext = 8192;
    window.setModels({spec});
    Conversation conversation{"a216e594-d78a-41f5-a9a5-90170838451b", "Verifica grafica", spec.id, {}, QDateTime::currentDateTimeUtc()};
    for (int index = 0; index < 12; ++index) {
        const QString content = index % 2 == 0
            ? QStringLiteral("Questo è un transcript sintetico per misurare il costo della finestra, senza eseguire un modello.")
            : QStringLiteral("La lettura deve restare chiara anche con codice e paragrafi.\n\n```cpp\nint main() { return 0; }\n```\n\nLa CPU del motore non è misurata in questo test.");
        conversation.messages.append(Message{index % 2 == 0 ? "user" : "assistant", content, false, {}});
    }
    window.setConversations({conversation}); window.showConversation(conversation);
    window.show();
    QTimer stream;
    QString text;
    int events = 0;
    QObject::connect(&stream, &QTimer::timeout, &window, [&] {
        text += QStringLiteral("Testo sintetico per verificare la finestra. ");
        window.setDraftResponse(text); ++events;
    });
    QElapsedTimer measured;
    ProcessCost before;
    QTimer::singleShot(600, &window, [&] {
        window.setBusy(scenario == "stream", "Misura grafica — nessun modello in esecuzione");
        before = processCost(); measured.start();
        if (scenario == "stream") stream.start(20);
        QTimer::singleShot(duration, &window, [&] {
            stream.stop();
            const auto after = processCost();
            const auto elapsed = measured.elapsed();
            const QJsonObject report{{"schema", "silicon-ui-cost-v1"}, {"synthetic_ui_only", true},
                {"scenario", scenario}, {"theme", theme}, {"elapsed_ms", static_cast<double>(elapsed)},
                {"cpu_ms", after.cpuMs - before.cpuMs}, {"private_bytes", after.privateBytes},
                {"working_set_bytes", after.workingSetBytes}, {"stream_events", events},
                {"visible_window", window.isVisible()}, {"native_platform", QGuiApplication::platformName()}};
            QFile output(parser.value("output"));
            if (!output.open(QIODevice::WriteOnly) || output.write(QJsonDocument(report).toJson()) < 0) { app.exit(3); return; }
            if (parser.isSet("capture") && !window.grab().save(parser.value("capture"), "PNG")) { app.exit(4); return; }
            app.exit(0);
        });
    });
    return app.exec();
}
