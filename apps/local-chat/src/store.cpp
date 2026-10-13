#include "store.h"

#include <QCryptographicHash>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QSet>
#include <QStandardPaths>
#include <QUuid>

#include <algorithm>
#include <utility>

namespace silicon {
namespace {

constexpr qint64 kConversationLimit = 32LL * 1024 * 1024;
constexpr qint64 kCatalogLimit = 4LL * 1024 * 1024;
constexpr qint64 kReceiptLimit = 1024LL * 1024;
constexpr qint64 kHashChunkSize = 1024LL * 1024;

void setError(QString *error, const QString &message)
{
    if (error)
        *error = message;
}

bool canonicalId(const QString &id)
{
    if (id.size() != 36)
        return false;
    const QUuid uuid(id);
    return !uuid.isNull() && uuid.toString(QUuid::WithoutBraces).compare(id, Qt::CaseInsensitive) == 0;
}

QString conversationPath(const QString &directory, const QString &id)
{
    return QDir(directory).filePath(QStringLiteral("conversations/%1.json").arg(id.toLower()));
}

QString conversationsDirectoryPath(const QString &directory)
{
    return QDir(directory).filePath(QStringLiteral("conversations"));
}

bool validMessageRole(const QString &role)
{
    return role == QStringLiteral("user") || role == QStringLiteral("assistant")
        || role == QStringLiteral("system");
}

QJsonObject conversationToJson(const Conversation &conversation)
{
    QJsonArray messages;
    for (const Message &message : conversation.messages) {
        messages.append(QJsonObject{
            {QStringLiteral("role"), message.role},
            {QStringLiteral("content"), message.content},
            {QStringLiteral("interrupted"), message.interrupted},
            {QStringLiteral("statistics"), message.statistics},
        });
    }

    return QJsonObject{
        {QStringLiteral("schemaVersion"), 1},
        {QStringLiteral("id"), conversation.id.toLower()},
        {QStringLiteral("title"), conversation.title},
        {QStringLiteral("modelId"), conversation.modelId},
        {QStringLiteral("updatedAt"), conversation.updatedAt.toUTC().toString(Qt::ISODateWithMs)},
        {QStringLiteral("messages"), messages},
    };
}

bool conversationFromJson(const QJsonObject &object, const QString &fileId,
                          Conversation *conversation, QString *reason)
{
    if (object.value(QStringLiteral("schemaVersion")).toInt(-1) != 1) {
        *reason = QStringLiteral("unsupported or missing schemaVersion (expected 1)");
        return false;
    }

    const QString id = object.value(QStringLiteral("id")).toString();
    if (!canonicalId(id) || id.compare(fileId, Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("conversation id is invalid or does not match the filename");
        return false;
    }

    const QJsonValue titleValue = object.value(QStringLiteral("title"));
    const QJsonValue modelValue = object.value(QStringLiteral("modelId"));
    const QJsonValue updatedValue = object.value(QStringLiteral("updatedAt"));
    const QJsonValue messagesValue = object.value(QStringLiteral("messages"));
    if (!titleValue.isString() || !modelValue.isString() || !updatedValue.isString()
        || !messagesValue.isArray()) {
        *reason = QStringLiteral("required conversation fields have the wrong type");
        return false;
    }

    const QDateTime updatedAt = QDateTime::fromString(updatedValue.toString(), Qt::ISODateWithMs);
    if (!updatedAt.isValid()) {
        *reason = QStringLiteral("updatedAt is not a valid ISO-8601 timestamp");
        return false;
    }

    Conversation parsed;
    parsed.id = id.toLower();
    parsed.title = titleValue.toString();
    parsed.modelId = modelValue.toString();
    parsed.updatedAt = updatedAt.toUTC();
    const QJsonArray messages = messagesValue.toArray();
    parsed.messages.reserve(messages.size());
    for (qsizetype index = 0; index < messages.size(); ++index) {
        const QJsonValue value = messages.at(index);
        if (!value.isObject()) {
            *reason = QStringLiteral("message %1 is not an object").arg(index);
            return false;
        }
        const QJsonObject messageObject = value.toObject();
        const QJsonValue role = messageObject.value(QStringLiteral("role"));
        const QJsonValue content = messageObject.value(QStringLiteral("content"));
        const QJsonValue interrupted = messageObject.value(QStringLiteral("interrupted"));
        const QJsonValue statistics = messageObject.value(QStringLiteral("statistics"));
        if (!role.isString() || !validMessageRole(role.toString()) || !content.isString()
            || !interrupted.isBool() || !statistics.isObject()) {
            *reason = QStringLiteral("message %1 has invalid role, content, interruption, or statistics").arg(index);
            return false;
        }
        parsed.messages.append(Message{role.toString(), content.toString(), interrupted.toBool(), statistics.toObject()});
    }

    *conversation = std::move(parsed);
    return true;
}

QString resolvePath(const QString &rawPath, const QString &catalogDirectory)
{
    if (rawPath.trimmed().isEmpty())
        return {};
    const QFileInfo supplied(rawPath);
    const QString absolute = supplied.isAbsolute()
        ? supplied.absoluteFilePath()
        : QDir(catalogDirectory).absoluteFilePath(rawPath);
    const QFileInfo resolved(absolute);
    const QString canonical = resolved.canonicalFilePath();
    return canonical.isEmpty() ? QDir::cleanPath(resolved.absoluteFilePath()) : canonical;
}

bool isRegularFile(const QString &path)
{
    const QFileInfo info(path);
    return !path.isEmpty() && info.exists() && info.isFile();
}

QString pathComparisonKey(const QString &path)
{
    return QDir::cleanPath(QFileInfo(path).absoluteFilePath()).toCaseFolded();
}

QString resolveArgumentFile(const QString &argument, const QString &workingDirectory)
{
    QString candidate = argument;
    if (candidate.startsWith(QLatin1Char('-'))) {
        const qsizetype equals = candidate.indexOf(QLatin1Char('='));
        if (equals < 0)
            return {};
        candidate = candidate.mid(equals + 1);
    }
    if (candidate.isEmpty())
        return {};
    const QFileInfo supplied(candidate);
    const QString absolute = supplied.isAbsolute()
        ? supplied.absoluteFilePath()
        : QDir(workingDirectory).absoluteFilePath(candidate);
    const QFileInfo info(absolute);
    return info.exists() && info.isFile() ? info.canonicalFilePath() : QString();
}

bool argumentLooksLikePythonScript(const QString &argument)
{
    QString candidate = argument;
    if (candidate.startsWith(QLatin1Char('-'))) {
        const qsizetype equals = candidate.indexOf(QLatin1Char('='));
        if (equals < 0)
            return false;
        candidate = candidate.mid(equals + 1);
    }
    return candidate.endsWith(QStringLiteral(".py"), Qt::CaseInsensitive);
}

bool isPythonExecutable(const QString &path)
{
    const QString name = QFileInfo(path).fileName();
    return name.startsWith(QStringLiteral("python"), Qt::CaseInsensitive)
        || path.endsWith(QStringLiteral(".py"), Qt::CaseInsensitive);
}

QJsonArray stringListToJson(const QStringList &values)
{
    QJsonArray array;
    for (const QString &value : values)
        array.append(value);
    return array;
}

bool isSha256(const QString &value)
{
    if (value.size() != 64)
        return false;
    for (const QChar character : value) {
        if (!character.isDigit() && !(character >= QLatin1Char('a') && character <= QLatin1Char('f'))
            && !(character >= QLatin1Char('A') && character <= QLatin1Char('F')))
            return false;
    }
    return true;
}

bool sha256File(const QString &path, QString *hash, QString *reason)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        *reason = QStringLiteral("cannot read %1: %2").arg(QFileInfo(path).fileName(), file.errorString());
        return false;
    }
    QCryptographicHash digest(QCryptographicHash::Sha256);
    while (!file.atEnd()) {
        const QByteArray chunk = file.read(kHashChunkSize);
        if (chunk.isEmpty() && !file.atEnd()) {
            *reason = QStringLiteral("cannot read more bytes while hashing %1: %2")
                .arg(QFileInfo(path).fileName(), file.errorString());
            return false;
        }
        digest.addData(chunk);
    }
    *hash = QString::fromLatin1(digest.result().toHex());
    return true;
}

bool receiptApproves(const QJsonObject &qualification, const QString &catalogDirectory,
                     const QString &executable, const QString &artifact, const QString &tokenizer,
                     const QString &python, const QString &workingDirectory, const QString &protocol,
                     int maxContext, int maxNewTokens, int timeoutSeconds, bool streaming,
                     const QStringList &arguments,
                     QString *reason)
{
    if (!qualification.value(QStringLiteral("approved")).isBool()
        || !qualification.value(QStringLiteral("approved")).toBool()) {
        *reason = QStringLiteral("catalog qualification is not approved");
        return false;
    }

    const QString executableHash = qualification.value(QStringLiteral("executable_sha256")).toString();
    const QString artifactHash = qualification.value(QStringLiteral("artifact_sha256")).toString();
    const QString tokenizerHash = qualification.value(QStringLiteral("tokenizer_sha256")).toString();
    if (!isSha256(executableHash) || !isSha256(artifactHash) || !isSha256(tokenizerHash)) {
        *reason = QStringLiteral("qualification must include valid executable_sha256, artifact_sha256, and tokenizer_sha256 values");
        return false;
    }
    QString receiptName = qualification.value(QStringLiteral("receipt")).toString();
    if (receiptName.isEmpty())
        receiptName = qualification.value(QStringLiteral("receipt_path")).toString();
    const QString receiptPath = resolvePath(receiptName, catalogDirectory);
    QFile receiptFile(receiptPath);
    if (receiptPath.isEmpty() || !receiptFile.open(QIODevice::ReadOnly)) {
        *reason = QStringLiteral("qualification receipt is missing or unreadable");
        return false;
    }
    if (receiptFile.size() > kReceiptLimit) {
        *reason = QStringLiteral("qualification receipt exceeds the 1 MiB limit");
        return false;
    }
    QJsonParseError parseError;
    const QByteArray receiptBytes = receiptFile.read(kReceiptLimit + 1);
    if (receiptBytes.size() > kReceiptLimit || receiptFile.error() != QFileDevice::NoError) {
        *reason = QStringLiteral("qualification receipt could not be read within the 1 MiB limit");
        return false;
    }
    const QJsonDocument receipt = QJsonDocument::fromJson(receiptBytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !receipt.isObject()) {
        *reason = QStringLiteral("qualification receipt is invalid JSON: %1").arg(parseError.errorString());
        return false;
    }
    const QJsonObject receiptObject = receipt.object();
    QString scope = receiptObject.value(QStringLiteral("scope")).toString();
    if (scope.isEmpty())
        scope = receiptObject.value(QStringLiteral("runtime_scope")).toString();
    scope.replace(QLatin1Char('_'), QLatin1Char('-'));
    if (!receiptObject.value(QStringLiteral("admitted")).isBool()
        || !receiptObject.value(QStringLiteral("admitted")).toBool()
        || scope.compare(QStringLiteral("cpu-runtime"), Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("receipt must admit the model for the cpu-runtime scope");
        return false;
    }
    if (receiptObject.value(QStringLiteral("executable_sha256")).toString().compare(executableHash, Qt::CaseInsensitive) != 0
        || receiptObject.value(QStringLiteral("artifact_sha256")).toString().compare(artifactHash, Qt::CaseInsensitive) != 0
        || receiptObject.value(QStringLiteral("tokenizer_sha256")).toString().compare(tokenizerHash, Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("receipt hashes do not match the catalog qualification hashes");
        return false;
    }
    const QJsonValue receiptContext = receiptObject.value(QStringLiteral("max_context"));
    const QJsonValue receiptMaxNew = receiptObject.value(QStringLiteral("max_new_tokens"));
    const QJsonValue receiptTimeout = receiptObject.value(QStringLiteral("timeout_seconds"));
    const QString receiptWorkingDirectory = resolvePath(
        receiptObject.value(QStringLiteral("working_directory")).toString(), catalogDirectory);
    if (receiptObject.value(QStringLiteral("protocol")).toString() != protocol
        || !receiptContext.isDouble() || receiptContext.toDouble() != maxContext
        || !receiptMaxNew.isDouble() || receiptMaxNew.toDouble() != maxNewTokens
        || !receiptTimeout.isDouble() || receiptTimeout.toDouble() != timeoutSeconds
        || receiptWorkingDirectory != workingDirectory
        || !receiptObject.value(QStringLiteral("streaming")).isBool()
        || receiptObject.value(QStringLiteral("streaming")).toBool() != streaming
        || receiptObject.value(QStringLiteral("arguments")) != stringListToJson(arguments)) {
        *reason = QStringLiteral("receipt runtime configuration does not match catalog protocol, limits, timeout, working directory, streaming, or arguments");
        return false;
    }

    const QJsonValue qualifiedFilesValue = qualification.value(QStringLiteral("files"));
    const QJsonValue receiptFilesValue = receiptObject.value(QStringLiteral("files"));
    if ((!qualifiedFilesValue.isUndefined() && !qualifiedFilesValue.isArray())
        || qualifiedFilesValue != receiptFilesValue) {
        *reason = QStringLiteral("receipt files must exactly match the catalog qualification files array");
        return false;
    }

    QSet<QString> boundPaths;
    QList<QPair<QString, QString>> qualifiedFiles;
    if (qualifiedFilesValue.isArray()) {
        for (const QJsonValue &fileValue : qualifiedFilesValue.toArray()) {
            if (!fileValue.isObject()) {
                *reason = QStringLiteral("qualification files entries must contain path and sha256");
                return false;
            }
            const QJsonObject fileObject = fileValue.toObject();
            const QString rawPath = fileObject.value(QStringLiteral("path")).toString();
            const QString hash = fileObject.value(QStringLiteral("sha256")).toString();
            const QString path = resolvePath(rawPath, catalogDirectory);
            if (rawPath.isEmpty() || !isSha256(hash) || !isRegularFile(path)) {
                *reason = QStringLiteral("qualification helper file has an invalid path, hash, or missing file");
                return false;
            }
            const QString key = pathComparisonKey(path);
            if (boundPaths.contains(key)) {
                *reason = QStringLiteral("qualification helper file paths must be unique");
                return false;
            }
            boundPaths.insert(key);
            qualifiedFiles.append(qMakePair(path, hash));
        }
    }

    const QStringList separateTokenizerFiles = {
        QStringLiteral("tokenizer_config.json"),
        QStringLiteral("chat_template.jinja"),
        QStringLiteral("chat_template.json"),
    };
    for (const QString &name : separateTokenizerFiles) {
        const QString sibling = QDir(QFileInfo(tokenizer).absolutePath()).filePath(name);
        const QFileInfo siblingInfo(sibling);
        if (siblingInfo.exists() && siblingInfo.isFile()
            && !boundPaths.contains(pathComparisonKey(siblingInfo.canonicalFilePath()))) {
            *reason = QStringLiteral("separate tokenizer configuration or chat template is not covered by qualification.files: %1")
                .arg(name);
            return false;
        }
    }
    if (!python.isEmpty() && !boundPaths.contains(pathComparisonKey(python))) {
        *reason = QStringLiteral("configured Python helper is not covered by qualification.files");
        return false;
    }

    const bool pythonRuntime = isPythonExecutable(executable);
    bool pythonScriptFound = !pythonRuntime;
    for (const QString &argument : arguments) {
        const QString argumentPath = resolveArgumentFile(argument, workingDirectory);
        if (!argumentPath.isEmpty()
            && pathComparisonKey(argumentPath) != pathComparisonKey(executable)
            && pathComparisonKey(argumentPath) != pathComparisonKey(artifact)
            && pathComparisonKey(argumentPath) != pathComparisonKey(tokenizer)
            && pathComparisonKey(argumentPath) != pathComparisonKey(python)
            && !boundPaths.contains(pathComparisonKey(argumentPath))) {
            *reason = QStringLiteral("runtime argument references an unbound helper file: %1")
                .arg(QFileInfo(argumentPath).fileName());
            return false;
        }
        if (pythonRuntime && argumentLooksLikePythonScript(argument)) {
            if (argumentPath.isEmpty()) {
                *reason = QStringLiteral("configured Python adapter script is missing");
                return false;
            }
            if (!boundPaths.contains(pathComparisonKey(argumentPath))) {
                *reason = QStringLiteral("configured Python adapter script is not covered by qualification.files");
                return false;
            }
            pythonScriptFound = true;
        }
    }
    if (!pythonScriptFound) {
        *reason = QStringLiteral("Python runtime requires an adapter script in arguments covered by qualification.files");
        return false;
    }

    QString actualHash;
    if (!sha256File(executable, &actualHash, reason))
        return false;
    if (actualHash.compare(executableHash, Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("executable SHA-256 does not match the qualification receipt");
        return false;
    }
    if (!sha256File(tokenizer, &actualHash, reason))
        return false;
    if (actualHash.compare(tokenizerHash, Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("tokenizer SHA-256 does not match the qualification receipt");
        return false;
    }
    for (const auto &qualifiedFile : qualifiedFiles) {
        if (!sha256File(qualifiedFile.first, &actualHash, reason))
            return false;
        if (actualHash.compare(qualifiedFile.second, Qt::CaseInsensitive) != 0) {
            *reason = QStringLiteral("qualified helper file SHA-256 mismatch: %1")
                .arg(QFileInfo(qualifiedFile.first).fileName());
            return false;
        }
    }
    if (!sha256File(artifact, &actualHash, reason))
        return false;
    if (actualHash.compare(artifactHash, Qt::CaseInsensitive) != 0) {
        *reason = QStringLiteral("model artifact SHA-256 does not match the qualification receipt");
        return false;
    }
    return true;
}

QString firstMissingPathReason(const ModelSpec &model)
{
    if (!isRegularFile(model.executable))
        return QStringLiteral("runtime executable is missing");
    if (!isRegularFile(model.artifact))
        return QStringLiteral("model artifact is missing");
    if (!isRegularFile(model.tokenizer))
        return QStringLiteral("tokenizer is missing");
    if (!model.python.isEmpty() && !isRegularFile(model.python))
        return QStringLiteral("configured Python interpreter is missing");
    if (model.workingDirectory.isEmpty() || !QFileInfo(model.workingDirectory).isDir())
        return QStringLiteral("working directory is missing");
    return {};
}

} // namespace

ConversationStore::ConversationStore(QString directory)
    : directory_(QDir::cleanPath(QFileInfo(std::move(directory)).absoluteFilePath()))
{
}

QList<Conversation> ConversationStore::load(QString *error) const
{
    if (error)
        error->clear();
    const QString conversationsDirectory = conversationsDirectoryPath(directory_);
    const QFileInfo directoryInfo(conversationsDirectory);
    if (!directoryInfo.exists())
        return {};
    if (!directoryInfo.isDir() || directoryInfo.isSymLink()) {
        setError(error, QStringLiteral("conversation path is not a directory: %1").arg(conversationsDirectory));
        return {};
    }

    QDir directory(conversationsDirectory);
    const QFileInfoList files = directory.entryInfoList({QStringLiteral("*.json")},
        QDir::Files | QDir::NoSymLinks, QDir::Name);
    QList<Conversation> conversations;
    QStringList errors;
    for (const QFileInfo &info : files) {
        const QString fileId = info.completeBaseName();
        if (!canonicalId(fileId)) {
            errors.append(QStringLiteral("%1: filename must be a canonical UUID").arg(info.fileName()));
            continue;
        }
        if (info.size() > kConversationLimit) {
            errors.append(QStringLiteral("%1: file exceeds the 32 MiB limit").arg(info.fileName()));
            continue;
        }
        QFile file(info.absoluteFilePath());
        if (!file.open(QIODevice::ReadOnly)) {
            errors.append(QStringLiteral("%1: cannot read file: %2").arg(info.fileName(), file.errorString()));
            continue;
        }
        const QByteArray bytes = file.read(kConversationLimit + 1);
        if (bytes.size() > kConversationLimit) {
            errors.append(QStringLiteral("%1: file exceeds the 32 MiB limit").arg(info.fileName()));
            continue;
        }
        if (file.error() != QFileDevice::NoError) {
            errors.append(QStringLiteral("%1: cannot read file: %2").arg(info.fileName(), file.errorString()));
            continue;
        }
        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(bytes, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            errors.append(QStringLiteral("%1: invalid JSON: %2").arg(info.fileName(), parseError.errorString()));
            continue;
        }
        Conversation conversation;
        QString reason;
        if (!conversationFromJson(document.object(), fileId, &conversation, &reason)) {
            errors.append(QStringLiteral("%1: %2").arg(info.fileName(), reason));
            continue;
        }
        conversations.append(std::move(conversation));
    }
    std::sort(conversations.begin(), conversations.end(), [](const Conversation &left, const Conversation &right) {
        return left.updatedAt > right.updatedAt;
    });
    if (!errors.isEmpty())
        setError(error, errors.join(QLatin1Char('\n')));
    return conversations;
}

bool ConversationStore::save(const Conversation &conversation, QString *error) const
{
    if (error)
        error->clear();
    if (!canonicalId(conversation.id)) {
        setError(error, QStringLiteral("conversation id must be a canonical UUID"));
        return false;
    }
    if (!conversation.updatedAt.isValid()) {
        setError(error, QStringLiteral("conversation updatedAt must be valid"));
        return false;
    }
    for (qsizetype index = 0; index < conversation.messages.size(); ++index) {
        if (!validMessageRole(conversation.messages.at(index).role)) {
            setError(error, QStringLiteral("message %1 has an unsupported role").arg(index));
            return false;
        }
    }

    const QByteArray bytes = QJsonDocument(conversationToJson(conversation)).toJson(QJsonDocument::Compact);
    if (bytes.size() > kConversationLimit) {
        setError(error, QStringLiteral("conversation exceeds the 32 MiB limit"));
        return false;
    }
    const QString conversationsDirectory = conversationsDirectoryPath(directory_);
    const QFileInfo existingDirectory(conversationsDirectory);
    if (existingDirectory.exists() && existingDirectory.isSymLink()) {
        setError(error, QStringLiteral("conversation directory must not be a symbolic link: %1").arg(conversationsDirectory));
        return false;
    }
    if (!QDir().mkpath(conversationsDirectory)) {
        setError(error, QStringLiteral("cannot create conversation directory: %1").arg(conversationsDirectory));
        return false;
    }
    QSaveFile file(conversationPath(directory_, conversation.id));
    if (!file.open(QIODevice::WriteOnly)) {
        setError(error, QStringLiteral("cannot open conversation for atomic save: %1").arg(file.errorString()));
        return false;
    }
    if (file.write(bytes) != bytes.size()) {
        setError(error, QStringLiteral("cannot write conversation: %1").arg(file.errorString()));
        file.cancelWriting();
        return false;
    }
    if (!file.commit()) {
        setError(error, QStringLiteral("cannot commit conversation: %1").arg(file.errorString()));
        return false;
    }
    return true;
}

bool ConversationStore::remove(const QString &id, QString *error) const
{
    if (error)
        error->clear();
    if (!canonicalId(id)) {
        setError(error, QStringLiteral("conversation id must be a canonical UUID"));
        return false;
    }
    const QFileInfo directoryInfo(conversationsDirectoryPath(directory_));
    if (directoryInfo.exists() && directoryInfo.isSymLink()) {
        setError(error, QStringLiteral("conversation directory must not be a symbolic link"));
        return false;
    }
    const QString path = conversationPath(directory_, id);
    if (!QFileInfo::exists(path))
        return true;
    if (!QFile::remove(path)) {
        setError(error, QStringLiteral("cannot remove conversation file: %1").arg(path));
        return false;
    }
    return true;
}

QString ConversationStore::defaultDirectory()
{
    const QString data = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    return QDir(data).filePath(QStringLiteral("local-chat"));
}

QList<ModelSpec> ModelCatalog::load(const QString &filePath, QString *error)
{
    if (error)
        error->clear();
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        setError(error, QStringLiteral("cannot read model catalog: %1").arg(file.errorString()));
        return {};
    }
    if (file.size() > kCatalogLimit) {
        setError(error, QStringLiteral("model catalog exceeds the 4 MiB limit"));
        return {};
    }
    QJsonParseError parseError;
    const QByteArray catalogBytes = file.read(kCatalogLimit + 1);
    if (catalogBytes.size() > kCatalogLimit || file.error() != QFileDevice::NoError) {
        setError(error, QStringLiteral("model catalog could not be read within the 4 MiB limit"));
        return {};
    }
    const QJsonDocument document = QJsonDocument::fromJson(catalogBytes, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        setError(error, QStringLiteral("model catalog is invalid JSON: %1").arg(parseError.errorString()));
        return {};
    }
    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("schemaVersion")).toInt(-1) != 1
        || !root.value(QStringLiteral("models")).isArray()) {
        setError(error, QStringLiteral("model catalog requires schemaVersion 1 and a models array"));
        return {};
    }
    const QString catalogDirectory = QFileInfo(filePath).absolutePath();
    const QJsonArray entries = root.value(QStringLiteral("models")).toArray();
    QList<ModelSpec> models;
    models.reserve(entries.size());
    QHash<QString, int> idCounts;
    for (const QJsonValue &value : entries) {
        if (value.isObject()) {
            const QString id = value.toObject().value(QStringLiteral("id")).toString();
            if (!id.isEmpty())
                ++idCounts[id];
        }
    }

    for (qsizetype index = 0; index < entries.size(); ++index) {
        const QJsonValue value = entries.at(index);
        ModelSpec model;
        QStringList reasons;
        if (!value.isObject()) {
            model.unavailableReason = QStringLiteral("model entry %1 is not an object").arg(index);
            models.append(std::move(model));
            continue;
        }
        const QJsonObject object = value.toObject();
        model.id = object.value(QStringLiteral("id")).toString();
        model.name = object.value(QStringLiteral("name")).toString();
        model.revision = object.value(QStringLiteral("revision")).toString();
        model.protocol = object.value(QStringLiteral("protocol")).toString();
        model.executable = resolvePath(object.value(QStringLiteral("executable")).toString(), catalogDirectory);
        model.artifact = resolvePath(object.value(QStringLiteral("artifact")).toString(), catalogDirectory);
        model.tokenizer = resolvePath(object.value(QStringLiteral("tokenizer")).toString(), catalogDirectory);
        model.python = resolvePath(object.value(QStringLiteral("python")).toString(), catalogDirectory);
        model.workingDirectory = resolvePath(object.value(QStringLiteral("working_directory")).toString(), catalogDirectory);
        model.qualification = object.value(QStringLiteral("qualification")).toObject();

        const QJsonValue arguments = object.value(QStringLiteral("arguments"));
        if (arguments.isUndefined()) {
            reasons.append(QStringLiteral("arguments is required"));
        } else if (!arguments.isArray()) {
            reasons.append(QStringLiteral("arguments must be an array of strings"));
        } else if (arguments.isArray()) {
            for (const QJsonValue &argument : arguments.toArray()) {
                if (!argument.isString()) {
                    reasons.append(QStringLiteral("arguments must contain only strings"));
                    break;
                }
                model.arguments.append(argument.toString());
            }
        }
        const QJsonValue streaming = object.value(QStringLiteral("streaming"));
        if (!streaming.isBool())
            reasons.append(QStringLiteral("streaming must be a boolean"));
        else
            model.streaming = streaming.toBool();
        const auto readBoundedInt = [&object, &reasons](const QString &key, int minimum, int maximum, int fallback) {
            const QJsonValue value = object.value(key);
            if (value.isUndefined()) {
                reasons.append(QStringLiteral("%1 is required").arg(key));
                return fallback;
            }
            if (!value.isDouble() || value.toDouble() != double(value.toInt(-1))
                || value.toInt(-1) < minimum || value.toInt(-1) > maximum) {
                reasons.append(QStringLiteral("%1 must be an integer between %2 and %3")
                    .arg(key).arg(minimum).arg(maximum));
                return fallback;
            }
            return value.toInt();
        };
        model.maxContext = readBoundedInt(QStringLiteral("max_context"), 1, 10000000, 0);
        model.maxNewTokens = readBoundedInt(QStringLiteral("max_new_tokens"), 1, 1000000, 0);
        model.timeoutSeconds = readBoundedInt(QStringLiteral("timeout_seconds"), 1, 3600, 0);

        if (model.id.isEmpty() || model.name.isEmpty() || model.revision.isEmpty())
            reasons.append(QStringLiteral("id, name, and revision are required"));
        if (idCounts.value(model.id) > 1)
            reasons.append(QStringLiteral("model id is duplicated"));
        if (model.maxContext > 0 && model.maxNewTokens > model.maxContext)
            reasons.append(QStringLiteral("max_new_tokens cannot exceed max_context"));

        if (model.protocol != QStringLiteral("silicon-chat-v1"))
            reasons.append(QStringLiteral("unsupported protocol; expected silicon-chat-v1"));
        const QString missingPath = firstMissingPathReason(model);
        if (!missingPath.isEmpty())
            reasons.append(missingPath);

        QString approvalReason;
        if (reasons.isEmpty()) {
            model.approved = receiptApproves(model.qualification, catalogDirectory,
                model.executable, model.artifact, model.tokenizer, model.python, model.workingDirectory,
                model.protocol, model.maxContext, model.maxNewTokens, model.timeoutSeconds, model.streaming,
                model.arguments, &approvalReason);
        } else {
            approvalReason = QStringLiteral("model fields, protocol, or configured paths are invalid");
        }
        if (!model.approved)
            reasons.append(approvalReason);
        model.available = model.approved && model.protocol == QStringLiteral("silicon-chat-v1")
            && missingPath.isEmpty() && reasons.isEmpty();
        model.unavailableReason = reasons.join(QStringLiteral("; "));
        models.append(std::move(model));
    }
    return models;
}

} // namespace silicon
