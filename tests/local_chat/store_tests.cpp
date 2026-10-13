#include "../../apps/local-chat/src/store.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QtTest>

using namespace silicon;

namespace {

bool writeFile(const QString &path, const QByteArray &contents)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QFile file(path);
    return file.open(QIODevice::WriteOnly) && file.write(contents) == contents.size();
}

QString sha256(const QByteArray &contents)
{
    return QString::fromLatin1(QCryptographicHash::hash(contents, QCryptographicHash::Sha256).toHex());
}

QJsonObject modelEntry(const QString &root, const QString &protocol, bool approved,
                       const QString &artifact = QStringLiteral("model.bin"))
{
    Q_UNUSED(root);
    return QJsonObject{
        {QStringLiteral("id"), QStringLiteral("fixture-model")},
        {QStringLiteral("name"), QStringLiteral("Fixture model")},
        {QStringLiteral("revision"), QStringLiteral("r1")},
        {QStringLiteral("protocol"), protocol},
        {QStringLiteral("executable"), QStringLiteral("runtime.exe")},
        {QStringLiteral("artifact"), artifact},
        {QStringLiteral("tokenizer"), QStringLiteral("tokenizer.json")},
        {QStringLiteral("python"), QString()},
        {QStringLiteral("working_directory"), QStringLiteral("work")},
        {QStringLiteral("arguments"), QJsonArray{QStringLiteral("--serve")}},
        {QStringLiteral("max_context"), 4096},
        {QStringLiteral("max_new_tokens"), 256},
        {QStringLiteral("timeout_seconds"), 120},
        {QStringLiteral("streaming"), false},
        {QStringLiteral("qualification"), QJsonObject{{QStringLiteral("approved"), approved}}},
    };
}

bool writeCatalog(const QString &path, const QJsonArray &models)
{
    return writeFile(path, QJsonDocument(QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("models"), models},
    }).toJson(QJsonDocument::Compact));
}

} // namespace

class StoreTests final : public QObject {
    Q_OBJECT

private slots:
    void conversationRoundTripPreservesInterruptionAndStatistics()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        ConversationStore store(temporary.path());
        Conversation expected;
        expected.id = QStringLiteral("3f08cfc9-660a-4a80-9f2b-8dcdb130cc0a");
        expected.title = QStringLiteral("Unicode — chat");
        expected.modelId = QStringLiteral("fixture-model");
        expected.updatedAt = QDateTime::fromString(QStringLiteral("2026-10-13T12:30:00.125Z"), Qt::ISODateWithMs);
        expected.messages = {
            Message{QStringLiteral("user"), QStringLiteral("Hi 🌱"), false, {}},
            Message{QStringLiteral("assistant"), QStringLiteral("Partial answer"), true,
                QJsonObject{{QStringLiteral("generated_tokens"), 7},
                            {QStringLiteral("duration_seconds"), 1.25},
                            {QStringLiteral("decode_tokens_per_second"), 5.6}}},
        };

        QString error;
        QVERIFY2(store.save(expected, &error), qPrintable(error));
        const QList<Conversation> loaded = store.load(&error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(loaded.size(), 1);
        QCOMPARE(loaded.front().id, expected.id);
        QCOMPARE(loaded.front().title, expected.title);
        QCOMPARE(loaded.front().modelId, expected.modelId);
        QCOMPARE(loaded.front().updatedAt, expected.updatedAt);
        QCOMPARE(loaded.front().messages.size(), 2);
        QCOMPARE(loaded.front().messages.at(0).content, QStringLiteral("Hi 🌱"));
        QVERIFY(loaded.front().messages.at(1).interrupted);
        QCOMPARE(loaded.front().messages.at(1).statistics.value(QStringLiteral("generated_tokens")).toInt(), 7);
        QCOMPARE(loaded.front().messages.at(1).statistics.value(QStringLiteral("decode_tokens_per_second")).toDouble(), 5.6);

        QVERIFY2(store.remove(expected.id, &error), qPrintable(error));
        QVERIFY(store.load().isEmpty());
    }

    void rejectsTraversalAndInvalidRoles()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        ConversationStore store(temporary.path());
        Conversation conversation;
        conversation.id = QStringLiteral("../outside");
        conversation.updatedAt = QDateTime::currentDateTimeUtc();
        QString error;
        QVERIFY(!store.save(conversation, &error));
        QVERIFY(error.contains(QStringLiteral("UUID")));
        QVERIFY(!store.remove(QStringLiteral("../../outside"), &error));

        conversation.id = QStringLiteral("3f08cfc9-660a-4a80-9f2b-8dcdb130cc0a");
        conversation.messages.append(Message{QStringLiteral("tool"), QStringLiteral("no"), false, {}});
        QVERIFY(!store.save(conversation, &error));
        QVERIFY(error.contains(QStringLiteral("role")));
    }

    void corruptedConversationIsReportedAndPreserved()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString conversationPath = QDir(temporary.path()).filePath(
            QStringLiteral("conversations/3f08cfc9-660a-4a80-9f2b-8dcdb130cc0a.json"));
        QVERIFY(writeFile(conversationPath, QByteArrayLiteral("{ broken json")));

        ConversationStore store(temporary.path());
        QString error;
        QVERIFY(store.load(&error).isEmpty());
        QVERIFY(error.contains(QStringLiteral("invalid JSON")));
        QVERIFY(QFileInfo::exists(conversationPath));
    }

    void emptyCatalogIsValid()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString path = QDir(temporary.path()).filePath(QStringLiteral("models.json"));
        QVERIFY(writeCatalog(path, {}));
        QString error;
        QVERIFY(ModelCatalog::load(path, &error).isEmpty());
        QVERIFY(error.isEmpty());
    }

    void catalogKeepsUnqualifiedMissingAndUnsupportedModelsUnavailable()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QString path = QDir(temporary.path()).filePath(QStringLiteral("models.json"));
        QJsonObject unapproved = modelEntry(temporary.path(), QStringLiteral("silicon-chat-v1"), false);
        unapproved.insert(QStringLiteral("id"), QStringLiteral("unapproved"));
        QJsonObject missingArtifact = modelEntry(temporary.path(), QStringLiteral("silicon-chat-v1"), true,
            QStringLiteral("absent.bin"));
        missingArtifact.insert(QStringLiteral("id"), QStringLiteral("missing"));
        QJsonObject unsupported = modelEntry(temporary.path(), QStringLiteral("legacy-wire"), false);
        unsupported.insert(QStringLiteral("id"), QStringLiteral("unsupported"));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("runtime.exe")), QByteArrayLiteral("runtime")));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("model.bin")), QByteArrayLiteral("model")));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("tokenizer.json")), QByteArrayLiteral("{}")));
        QVERIFY(QDir().mkpath(QDir(temporary.path()).filePath(QStringLiteral("work"))));
        QVERIFY(writeCatalog(path, {unapproved, missingArtifact, unsupported}));

        QString error;
        const QList<ModelSpec> models = ModelCatalog::load(path, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(models.size(), 3);
        QVERIFY(!models.at(0).approved);
        QVERIFY(!models.at(0).available);
        QVERIFY(models.at(0).unavailableReason.contains(QStringLiteral("not approved")));
        QVERIFY(!models.at(1).available);
        QVERIFY(models.at(1).unavailableReason.contains(QStringLiteral("missing")));
        QVERIFY(!models.at(2).available);
        QVERIFY(models.at(2).unavailableReason.contains(QStringLiteral("unsupported protocol")));
    }

    void approvedCatalogRequiresMatchingReceiptAndFileHashes()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        const QByteArray executableBytes = QByteArrayLiteral("fake runtime fixture");
        const QByteArray artifactBytes = QByteArrayLiteral("fake packed model fixture");
        const QByteArray tokenizerBytes = QByteArrayLiteral("{\"fixture\":true}");
        const QByteArray tokenizerConfigBytes = QByteArrayLiteral("{\"bos_token\":\"<s>\"}");
        const QByteArray templateBytes = QByteArrayLiteral("{{ messages }}");
        const QByteArray adapterBytes = QByteArrayLiteral("print('adapter fixture')\n");
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("python.exe")), executableBytes));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("model.bin")), artifactBytes));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("tokenizer.json")), tokenizerBytes));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("tokenizer_config.json")), tokenizerConfigBytes));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("chat_template.jinja")), templateBytes));
        QVERIFY(QDir().mkpath(QDir(temporary.path()).filePath(QStringLiteral("work"))));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("work/adapter.py")), adapterBytes));

        const QString executableHash = sha256(executableBytes);
        const QString artifactHash = sha256(artifactBytes);
        const QString tokenizerHash = sha256(tokenizerBytes);
        const QString tokenizerConfigHash = sha256(tokenizerConfigBytes);
        const QString templateHash = sha256(templateBytes);
        const QString adapterHash = sha256(adapterBytes);
        const QString adapterPath = QDir(temporary.path()).filePath(QStringLiteral("work/adapter.py"));
        const QJsonArray arguments{adapterPath, QStringLiteral("--serve")};
        const QJsonArray files{
            QJsonObject{
                {QStringLiteral("path"), QStringLiteral("work/adapter.py")},
                {QStringLiteral("sha256"), adapterHash},
            },
            QJsonObject{
                {QStringLiteral("path"), QStringLiteral("tokenizer_config.json")},
                {QStringLiteral("sha256"), tokenizerConfigHash},
            },
            QJsonObject{
                {QStringLiteral("path"), QStringLiteral("chat_template.jinja")},
                {QStringLiteral("sha256"), templateHash},
            },
        };
        const QJsonObject receipt{
            {QStringLiteral("admitted"), true},
            {QStringLiteral("scope"), QStringLiteral("cpu-runtime")},
            {QStringLiteral("executable_sha256"), executableHash},
            {QStringLiteral("artifact_sha256"), artifactHash},
            {QStringLiteral("tokenizer_sha256"), tokenizerHash},
            {QStringLiteral("protocol"), QStringLiteral("silicon-chat-v1")},
            {QStringLiteral("max_context"), 4096},
            {QStringLiteral("max_new_tokens"), 256},
            {QStringLiteral("timeout_seconds"), 120},
            {QStringLiteral("working_directory"), QStringLiteral("work")},
            {QStringLiteral("streaming"), true},
            {QStringLiteral("arguments"), arguments},
            {QStringLiteral("files"), files},
        };
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("qualification.json")),
            QJsonDocument(receipt).toJson(QJsonDocument::Compact)));
        QJsonObject entry = modelEntry(temporary.path(), QStringLiteral("silicon-chat-v1"), true);
        entry.insert(QStringLiteral("executable"), QStringLiteral("python.exe"));
        entry.insert(QStringLiteral("arguments"), arguments);
        entry.insert(QStringLiteral("streaming"), true);
        entry.insert(QStringLiteral("qualification"), QJsonObject{
            {QStringLiteral("approved"), true},
            {QStringLiteral("receipt"), QStringLiteral("qualification.json")},
            {QStringLiteral("executable_sha256"), executableHash},
            {QStringLiteral("artifact_sha256"), artifactHash},
            {QStringLiteral("tokenizer_sha256"), tokenizerHash},
            {QStringLiteral("files"), files},
        });
        const QString path = QDir(temporary.path()).filePath(QStringLiteral("models.json"));
        QVERIFY(writeCatalog(path, {entry}));

        QString error;
        QList<ModelSpec> models = ModelCatalog::load(path, &error);
        QVERIFY2(error.isEmpty(), qPrintable(error));
        QCOMPARE(models.size(), 1);
        QVERIFY2(models.front().approved, qPrintable(models.front().unavailableReason));
        QVERIFY2(models.front().available, qPrintable(models.front().unavailableReason));
        QCOMPARE(models.front().executable, QFileInfo(QDir(temporary.path()).filePath(QStringLiteral("python.exe"))).canonicalFilePath());
        QCOMPARE(models.front().artifact, QFileInfo(QDir(temporary.path()).filePath(QStringLiteral("model.bin"))).canonicalFilePath());

        QVERIFY(QDir().mkpath(QDir(temporary.path()).filePath(QStringLiteral("other-work"))));
        QJsonObject alteredWorkingDirectory = entry;
        alteredWorkingDirectory.insert(QStringLiteral("working_directory"), QStringLiteral("other-work"));
        QVERIFY(writeCatalog(path, {alteredWorkingDirectory}));
        models = ModelCatalog::load(path, &error);
        QCOMPARE(models.size(), 1);
        QVERIFY(!models.front().approved);
        QVERIFY(models.front().unavailableReason.contains(QStringLiteral("working directory")));

        QJsonObject alteredTimeout = entry;
        alteredTimeout.insert(QStringLiteral("timeout_seconds"), 121);
        QVERIFY(writeCatalog(path, {alteredTimeout}));
        models = ModelCatalog::load(path, &error);
        QCOMPARE(models.size(), 1);
        QVERIFY(!models.front().approved);
        QVERIFY(models.front().unavailableReason.contains(QStringLiteral("timeout")));

        QVERIFY(writeCatalog(path, {entry}));
        models = ModelCatalog::load(path, &error);
        QVERIFY2(models.front().approved && models.front().available, qPrintable(models.front().unavailableReason));

        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("work/adapter.py")), QByteArrayLiteral("changed adapter")));
        models = ModelCatalog::load(path, &error);
        QCOMPARE(models.size(), 1);
        QVERIFY(!models.front().approved);
        QVERIFY(models.front().unavailableReason.contains(QStringLiteral("helper file SHA-256 mismatch")));
        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("work/adapter.py")), adapterBytes));

        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("tokenizer.json")), QByteArrayLiteral("changed tokenizer")));
        models = ModelCatalog::load(path, &error);
        QCOMPARE(models.size(), 1);
        QVERIFY(!models.front().approved);
        QVERIFY(!models.front().available);
        QVERIFY(models.front().unavailableReason.contains(QStringLiteral("SHA-256")));

        QVERIFY(writeFile(QDir(temporary.path()).filePath(QStringLiteral("tokenizer.json")), tokenizerBytes));
        entry.insert(QStringLiteral("arguments"), QJsonArray{QStringLiteral("adapter.py"), QStringLiteral("--changed")});
        QVERIFY(writeCatalog(path, {entry}));
        models = ModelCatalog::load(path, &error);
        QCOMPARE(models.size(), 1);
        QVERIFY(!models.front().approved);
        QVERIFY(models.front().unavailableReason.contains(QStringLiteral("runtime configuration")));
    }
};

QTEST_GUILESS_MAIN(StoreTests)
#include "store_tests.moc"
