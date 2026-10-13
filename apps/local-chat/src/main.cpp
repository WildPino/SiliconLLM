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
    QCoreApplication::setApplicationVersion("0.1.0");
    QCommandLineParser parser;
    parser.setApplicationDescription("Chat desktop locale per SiliconLLM");
    parser.addHelpOption(); parser.addVersionOption();
    parser.addOption({"catalog", "Catalogo locale degli artefatti qualificati", "file"});
    parser.addOption({"data-dir", "Directory delle conversazioni", "directory"});
    parser.addOption({"capture", "Salva una schermata della finestra e termina (verifica UI)", "png"});
    parser.process(app);
    MainWindow window;
    AppController controller(window, parser.isSet("data-dir") ? parser.value("data-dir") : ConversationStore::defaultDirectory());
    window.show();
    const auto catalog = parser.isSet("catalog") ? parser.value("catalog") : QSettings().value("catalog").toString();
    if (!catalog.isEmpty()) controller.loadCatalog(catalog);
    if (parser.isSet("capture")) QTimer::singleShot(350, &window, [&] {
        app.exit(window.grab().save(parser.value("capture"), "PNG") ? 0 : 2);
    });
    return app.exec();
}
