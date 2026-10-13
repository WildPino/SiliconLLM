#include "controller.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QSettings>
#include <QTimer>

using namespace silicon;

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName("SiliconLLM");
    QCoreApplication::setOrganizationName("SiliconLLM");
    QCoreApplication::setApplicationVersion("0.2.1");
    QCommandLineParser parser;
    parser.setApplicationDescription("Local desktop chat for SiliconLLM");
    parser.addHelpOption(); parser.addVersionOption();
    parser.addOption({"catalog", "Local catalog of qualified artifacts", "file"});
    parser.addOption({"data-dir", "Conversation directory", "directory"});
    parser.addOption({"capture", "Save a window screenshot and exit (UI verification)", "png"});
    parser.addOption({"theme", "Window theme: light or dark", "name"});
    parser.addOption({"capture-settings", "Save a settings screenshot and exit (UI verification)", "png"});
    parser.process(app);
    MainWindow window;
    if (parser.isSet("theme")) {
        if (parser.value("theme") != "light" && parser.value("theme") != "dark") return 2;
        window.setTheme(parser.value("theme"));
    }
    AppController controller(window, parser.isSet("data-dir") ? parser.value("data-dir") : ConversationStore::defaultDirectory());
    window.show();
    const auto catalog = parser.isSet("catalog") ? parser.value("catalog") : QSettings().value("catalog").toString();
    if (!catalog.isEmpty()) controller.loadCatalog(catalog);
    if (parser.isSet("capture")) QTimer::singleShot(350, &window, [&] {
        app.exit(window.grab().save(parser.value("capture"), "PNG") ? 0 : 2);
    });
    if (parser.isSet("capture-settings")) {
        QTimer::singleShot(350, &window, [&] { window.showSettingsDialog(); });
        QTimer::singleShot(550, &window, [&] {
            auto* dialog = QApplication::activeModalWidget();
            const bool saved = dialog && dialog->grab().save(parser.value("capture-settings"), "PNG");
            if (dialog) dialog->close();
            app.exit(saved ? 0 : 2);
        });
    }
    return app.exec();
}
