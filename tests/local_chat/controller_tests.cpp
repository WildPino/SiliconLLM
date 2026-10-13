#include "../../apps/local-chat/src/controller.h"

#include <QComboBox>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QtMath>
#include <QtTest>

#ifndef LOCAL_CHAT_FIXTURE
#error LOCAL_CHAT_FIXTURE must point to fixture_engine.py
#endif
#ifndef LOCAL_CHAT_PYTHON
#error LOCAL_CHAT_PYTHON must point to the Python interpreter
#endif

using namespace silicon;

namespace {
const QString fixturePath = QStringLiteral(LOCAL_CHAT_FIXTURE);
const QString pythonPath = QStringLiteral(LOCAL_CHAT_PYTHON);

bool writeFile(const QString& path, const QByteArray& bytes)
{
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size();
}

QString fileHash(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray chunk = file.read(1024 * 1024);
        if (chunk.isEmpty() && file.error() != QFileDevice::NoError) return {};
        hash.addData(chunk);
    }
    return QString::fromLatin1(hash.result().toHex());
}

QJsonArray fileBindings()
{
    return {{QJsonObject{{"path", QFileInfo(fixturePath).absoluteFilePath()},
                         {"sha256", fileHash(fixturePath)}}}};
}

bool writeCatalog(const QString& directory, const QList<QPair<QString, QString>>& fixtures)
{
    const QString executableHash = fileHash(pythonPath);
    if (executableHash.isEmpty() || fileHash(fixturePath).isEmpty()) return false;
    const QByteArray artifactBytes("synthetic fixture weights\n");
    const QByteArray tokenizerBytes("{\"fixture\":true}\n");
    const QString artifactPath = QDir(directory).filePath(QStringLiteral("fixture.bin"));
    const QString tokenizerPath = QDir(directory).filePath(QStringLiteral("tokenizer.json"));
    if (!writeFile(artifactPath, artifactBytes) || !writeFile(tokenizerPath, tokenizerBytes)) return false;
    const QString artifactHash = QString::fromLatin1(QCryptographicHash::hash(artifactBytes, QCryptographicHash::Sha256).toHex());
    const QString tokenizerHash = QString::fromLatin1(QCryptographicHash::hash(tokenizerBytes, QCryptographicHash::Sha256).toHex());
    QJsonArray models;
    for (const auto& entry : fixtures) {
        const QString id = entry.first;
        const QString mode = entry.second;
        const QStringList arguments = {QStringLiteral("-u"), QFileInfo(fixturePath).absoluteFilePath(),
                                       QStringLiteral("--mode=%1").arg(mode), QStringLiteral("--max-context=4096")};
        const QString receiptName = id + QStringLiteral("-receipt.json");
        const QJsonArray files = fileBindings();
        const QJsonObject receipt{
            {"admitted", true}, {"scope", "cpu-runtime"},
            {"executable_sha256", executableHash}, {"artifact_sha256", artifactHash},
            {"tokenizer_sha256", tokenizerHash}, {"protocol", "silicon-chat-v1"},
            {"max_context", 4096}, {"max_new_tokens", 32}, {"streaming", true},
            {"arguments", QJsonArray::fromStringList(arguments)}, {"files", files},
            {"working_directory", QFileInfo(fixturePath).absolutePath()}, {"timeout_seconds", 5}};
        if (!writeFile(QDir(directory).filePath(receiptName), QJsonDocument(receipt).toJson(QJsonDocument::Compact))) return false;
        const QJsonObject qualification{
            {"approved", true}, {"receipt", receiptName},
            {"executable_sha256", executableHash}, {"artifact_sha256", artifactHash},
            {"tokenizer_sha256", tokenizerHash}, {"files", files}};
        models.append(QJsonObject{
            {"id", id}, {"name", id}, {"revision", "fixture-r1"},
            {"protocol", "silicon-chat-v1"}, {"executable", pythonPath},
            {"artifact", artifactPath}, {"tokenizer", tokenizerPath}, {"python", ""},
            {"working_directory", QFileInfo(fixturePath).absolutePath()},
            {"arguments", QJsonArray::fromStringList(arguments)},
            {"max_context", 4096}, {"max_new_tokens", 32},
            {"timeout_seconds", 5}, {"streaming", true}, {"qualification", qualification}});
    }
    return writeFile(QDir(directory).filePath(QStringLiteral("catalog.json")),
                     QJsonDocument(QJsonObject{{"schemaVersion", 1}, {"models", models}}).toJson(QJsonDocument::Compact));
}

QString catalogPath(const QString& directory) { return QDir(directory).filePath(QStringLiteral("catalog.json")); }
QList<Conversation> loadConversations(const QString& directory)
{
    QString error;
    const auto chats = ConversationStore(directory).load(&error);
    if (!error.isEmpty()) qWarning().noquote() << error;
    return chats;
}

QListWidget* conversationList(MainWindow& window) { return window.findChild<QListWidget*>(QStringLiteral("conversationList")); }
QComboBox* modelSelector(MainWindow& window) { return window.findChild<QComboBox*>(QStringLiteral("modelSelector")); }
QLabel* statusLabel(MainWindow& window) { return window.findChild<QLabel*>(QStringLiteral("statusLabel")); }
}

class ControllerTests final : public QObject {
    Q_OBJECT
private slots:
    void twoTurnsPersistAndReloadWithDisplayedFirstContentTime()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString dataDirectory = QDir(temporary.path()).filePath(QStringLiteral("data"));
        QVERIFY(writeCatalog(temporary.path(), {{QStringLiteral("fixture-normal"), QStringLiteral("normal")}}));
        QString conversationId;
        double firstDisplayedSeconds = -1.0;
        {
            MainWindow window;
            AppController controller(window, dataDirectory);
            QSignalSpy displayed(&window, &MainWindow::firstContentDisplayed);
            window.show();
            controller.loadCatalog(catalogPath(temporary.path()));
            auto* selector = modelSelector(window);
            QVERIFY(selector);
            QTRY_COMPARE_WITH_TIMEOUT(selector->count(), 1, 5000);
            emit window.newConversationRequested(QStringLiteral("fixture-normal"));
            auto* list = conversationList(window);
            QVERIFY(list);
            QTRY_COMPARE(list->count(), 1);
            conversationId = list->item(0)->data(Qt::UserRole).toString();
            emit window.sendRequested(QStringLiteral("prima domanda"));
            QTRY_VERIFY_WITH_TIMEOUT([&] {
                const auto chats = loadConversations(dataDirectory);
                return !chats.isEmpty() && chats.first().id == conversationId && chats.first().messages.size() == 2;
            }(), 5000);
            QCOMPARE(displayed.count(), 1);
            auto chats = loadConversations(dataDirectory);
            QCOMPARE(chats.first().messages.at(1).content, QStringLiteral("Ciao 🌱 — café"));
            const auto firstStats = chats.first().messages.at(1).statistics;
            QVERIFY(firstStats.value(QStringLiteral("first_token_seconds")).isDouble());
            QVERIFY(firstStats.value(QStringLiteral("total_seconds")).isDouble());
            firstDisplayedSeconds = firstStats.value(QStringLiteral("first_token_seconds")).toDouble();
            QVERIFY(firstStats.value(QStringLiteral("total_seconds")).toDouble() - firstDisplayedSeconds >= 0.05);

            emit window.sendRequested(QStringLiteral("seconda domanda"));
            QTRY_VERIFY_WITH_TIMEOUT([&] {
                const auto saved = loadConversations(dataDirectory);
                return !saved.isEmpty() && saved.first().messages.size() == 4;
            }(), 5000);
            chats = loadConversations(dataDirectory);
            QCOMPARE(chats.first().messages.at(2).content, QStringLiteral("seconda domanda"));
            QCOMPARE(chats.first().messages.at(3).content, QStringLiteral("Ciao 🌱 — café"));
            QCOMPARE(displayed.count(), 2);
        }
        MainWindow restoredWindow;
        AppController restored(restoredWindow, dataDirectory);
        const auto persisted = loadConversations(dataDirectory);
        QCOMPARE(persisted.size(), 1);
        QCOMPARE(persisted.first().id, conversationId);
        QCOMPARE(persisted.first().messages.size(), 4);
        QCOMPARE(persisted.first().messages.at(0).content, QStringLiteral("prima domanda"));
        QCOMPARE(persisted.first().messages.at(2).content, QStringLiteral("seconda domanda"));
        QVERIFY(qAbs(persisted.first().messages.at(1).statistics.value(QStringLiteral("first_token_seconds")).toDouble()
                     - firstDisplayedSeconds) < 0.000001);
    }

    void modelSwitchCreatesIsolatedConversationState()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString dataDirectory = QDir(temporary.path()).filePath(QStringLiteral("data"));
        QVERIFY(writeCatalog(temporary.path(), {{QStringLiteral("model-a"), QStringLiteral("normal")},
                                                {QStringLiteral("model-b"), QStringLiteral("normal")}}));
        MainWindow window;
        AppController controller(window, dataDirectory);
        window.show();
        controller.loadCatalog(catalogPath(temporary.path()));
        QTRY_COMPARE_WITH_TIMEOUT(modelSelector(window)->count(), 2, 5000);
        emit window.newConversationRequested(QStringLiteral("model-a"));
        auto* list = conversationList(window);
        QTRY_COMPARE(list->count(), 1);
        const QString firstId = list->item(0)->data(Qt::UserRole).toString();
        emit window.sendRequested(QStringLiteral("solo A"));
        QTRY_VERIFY_WITH_TIMEOUT([&] {
            const auto chats = loadConversations(dataDirectory);
            return !chats.isEmpty() && chats.first().messages.size() == 2;
        }(), 5000);
        emit window.modelChangeRequested(QStringLiteral("model-b"));
        QTRY_COMPARE(list->count(), 2);
        const QString secondId = list->currentItem()->data(Qt::UserRole).toString();
        QVERIFY(secondId != firstId);
        emit window.sendRequested(QStringLiteral("solo B"));
        QTRY_VERIFY_WITH_TIMEOUT([&] {
            const auto chats = loadConversations(dataDirectory);
            if (chats.size() != 2) return false;
            for (const auto& chat : chats) if (chat.id == secondId) return chat.messages.size() == 2;
            return false;
        }(), 5000);
        const auto chats = loadConversations(dataDirectory);
        QCOMPARE(chats.size(), 2);
        for (const auto& chat : chats) {
            if (chat.id == firstId) {
                QCOMPARE(chat.modelId, QStringLiteral("model-a"));
                QCOMPARE(chat.messages.first().content, QStringLiteral("solo A"));
            } else {
                QCOMPARE(chat.id, secondId);
                QCOMPARE(chat.modelId, QStringLiteral("model-b"));
                QCOMPARE(chat.messages.first().content, QStringLiteral("solo B"));
            }
        }
    }

    void stopPersistsInterruptedPartialAssistant()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString dataDirectory = QDir(temporary.path()).filePath(QStringLiteral("data"));
        QVERIFY(writeCatalog(temporary.path(), {{QStringLiteral("slow-model"), QStringLiteral("slow")}}));
        MainWindow window;
        AppController controller(window, dataDirectory);
        window.show();
        QSignalSpy displayed(&window, &MainWindow::firstContentDisplayed);
        controller.loadCatalog(catalogPath(temporary.path()));
        QTRY_COMPARE_WITH_TIMEOUT(modelSelector(window)->count(), 1, 5000);
        emit window.newConversationRequested(QStringLiteral("slow-model"));
        emit window.sendRequested(QStringLiteral("ferma la risposta"));
        QTRY_VERIFY_WITH_TIMEOUT(displayed.count() > 0, 5000);
        emit window.stopRequested();
        QTRY_VERIFY_WITH_TIMEOUT([&] {
            const auto chats = loadConversations(dataDirectory);
            return !chats.isEmpty() && chats.first().messages.size() == 2;
        }(), 5000);
        const auto chat = loadConversations(dataDirectory).first();
        QVERIFY(!chat.messages.at(1).content.isEmpty());
        QVERIFY(chat.messages.at(1).interrupted);
        QCOMPARE(chat.messages.at(1).statistics.value(QStringLiteral("termination")).toString(), QStringLiteral("interrupted"));
    }

    void engineFailureWithoutPartialDoesNotInventAssistantMessage()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString dataDirectory = QDir(temporary.path()).filePath(QStringLiteral("data"));
        QVERIFY(writeCatalog(temporary.path(), {{QStringLiteral("broken-model"), QStringLiteral("invalid_line")}}));
        MainWindow window;
        AppController controller(window, dataDirectory);
        window.show();
        controller.loadCatalog(catalogPath(temporary.path()));
        QTRY_COMPARE_WITH_TIMEOUT(modelSelector(window)->count(), 1, 5000);
        emit window.newConversationRequested(QStringLiteral("broken-model"));
        emit window.sendRequested(QStringLiteral("domanda senza risposta"));
        QTRY_VERIFY_WITH_TIMEOUT(statusLabel(window)->property("error").toBool(), 5000);
        QVERIFY(!statusLabel(window)->text().trimmed().isEmpty());
        const auto chats = loadConversations(dataDirectory);
        QCOMPARE(chats.size(), 1);
        QCOMPARE(chats.first().messages.size(), 1);
        QCOMPARE(chats.first().messages.first().role, QStringLiteral("user"));
        QCOMPARE(chats.first().messages.first().content, QStringLiteral("domanda senza risposta"));
    }
};

QTEST_MAIN(ControllerTests)
#include "controller_tests.moc"
