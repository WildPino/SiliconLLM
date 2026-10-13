#include "engine.h"
#include <QDateTime>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QElapsedTimer>
#include <QUuid>
#include <cmath>
#ifdef Q_OS_WIN
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#endif

namespace silicon {
namespace {
constexpr qsizetype MaxLine = 1024 * 1024;
constexpr qsizetype MaxResponse = 16 * 1024 * 1024;
qint64 monotonicMilliseconds() {
    static const auto start = [] { QElapsedTimer t; t.start(); return t; }();
    return start.elapsed();
}
bool validMetric(const QJsonObject& stats, const char* name) {
    const auto value = stats.value(QLatin1String(name));
    return value.isUndefined() || value.isNull() ||
           (value.isDouble() && std::isfinite(value.toDouble()) && value.toDouble() >= 0);
}
void sampleProcessMemory(QJsonObject& stats, qint64 pid) {
#ifdef Q_OS_WIN
    const HANDLE handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, static_cast<DWORD>(pid));
    if (!handle) return;
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(handle, &counters, sizeof(counters)))
        stats.insert("working_set_bytes", static_cast<double>(counters.WorkingSetSize));
    CloseHandle(handle);
#else
    Q_UNUSED(stats); Q_UNUSED(pid);
#endif
}
}

EngineSession::EngineSession(QObject* parent) : QObject(parent) {
    deadline_.setSingleShot(true);
    connect(&deadline_, &QTimer::timeout, this, [this] { fail(tr("Engine request timed out.")); });
    connect(&process_, &QProcess::readyReadStandardOutput, this, &EngineSession::readOutput);
    connect(&process_, &QProcess::readyReadStandardError, this, &EngineSession::readErrors);
    connect(&process_, &QProcess::started, this, [this] {
#ifdef Q_OS_WIN
        const HANDLE handle = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(process_.processId()));
        const bool assigned = handle && AssignProcessToJobObject(static_cast<HANDLE>(processJob_), handle);
        if (handle) CloseHandle(handle);
        if (!assigned) fail(tr("Cannot guarantee termination of engine processes."));
#endif
    });
    connect(&process_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError error) {
        if (!stopping_ && error != QProcess::Crashed) fail(tr("Cannot communicate with the engine: %1").arg(process_.errorString()));
    });
    connect(&process_, qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            [this](int code, QProcess::ExitStatus) {
        if (stopping_) return;
        readOutput(); readErrors(); ready_ = false;
        if (busy()) fail(tr("Engine exited (code %1). %2").arg(code).arg(QString::fromUtf8(errors_)));
    });
}

EngineSession::~EngineSession() { shutdown(); }

bool EngineSession::generate(const ModelSpec& model, const Conversation& conversation, int maxNewTokens) {
    if (busy()) { emit failed(tr("A generation is already running.")); return false; }
    if (!model.available || !model.approved || model.protocol != QStringLiteral("silicon-chat-v1")) {
        emit failed(tr("This model has no approved, available CPU runtime.")); return false;
    }
    if (maxNewTokens < 1 || maxNewTokens > model.maxNewTokens || conversation.messages.isEmpty()) {
        emit failed(tr("Invalid request parameters.")); return false;
    }
    if (model_.id != model.id || process_.state() == QProcess::NotRunning) {
        stopProcess(); model_ = model; ready_ = false;
    }
    response_.clear(); streamed_ = false; generated_ = 0; firstTokenMs_ = -1;
    requestId_ = QUuid::createUuid().toString(QUuid::WithoutBraces);
    requestStarted_ = monotonicMilliseconds(); requestedLimit_ = maxNewTokens;
    QJsonArray messages;
    for (const auto& message : conversation.messages) {
        messages.append(QJsonObject{{"role", message.role}, {"content", message.content}});
    }
    pending_ = QJsonObject{{"protocol", "silicon-chat-v1"}, {"type", "generate"},
                          {"request_id", requestId_}, {"conversation_id", conversation.id},
                          {"reset", true}, {"messages", messages}, {"max_new_tokens", maxNewTokens}};
    deadline_.start(model.timeoutSeconds * 1000);
    if (process_.state() == QProcess::NotRunning) {
#ifdef Q_OS_WIN
        processJob_ = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!processJob_ || !SetInformationJobObject(static_cast<HANDLE>(processJob_), JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            fail(tr("Cannot create the engine process group.")); return false;
        }
#endif
        output_.clear(); errors_.clear();
        process_.setProgram(model.executable);
        QStringList arguments = model.arguments;
        for (auto& argument : arguments) {
            argument.replace("{artifact}", model.artifact);
            argument.replace("{tokenizer}", model.tokenizer);
        }
        process_.setArguments(arguments);
        process_.setWorkingDirectory(model.workingDirectory);
        process_.setProcessChannelMode(QProcess::SeparateChannels);
        auto environment = QProcessEnvironment::systemEnvironment();
        environment.insert("PYTHONUTF8", "1");
        process_.setProcessEnvironment(environment);
        emit statusChanged(tr("Loading model…"));
        process_.start();
    } else if (ready_) sendPending();
    return true;
}

void EngineSession::sendPending() {
    if (!busy() || pending_.isEmpty()) return;
    const QByteArray bytes = QJsonDocument(pending_).toJson(QJsonDocument::Compact) + '\n';
    if (bytes.size() > MaxResponse || process_.write(bytes) != bytes.size()) {
        fail(tr("Failed to send the request to the engine.")); return;
    }
    pending_ = {};
    emit statusChanged(tr("Preparing prompt and generating…"));
}

void EngineSession::readErrors() {
    errors_.append(process_.readAllStandardError());
    if (errors_.size() > 16384) errors_ = errors_.right(16384);
}

void EngineSession::readOutput() {
    output_.append(process_.readAllStandardOutput());
    while (true) {
        const auto end = output_.indexOf('\n');
        if (end < 0) break;
        if (end > MaxLine) { fail(tr("Engine event exceeds the size limit.")); return; }
        const auto line = output_.left(end); output_.remove(0, end + 1);
        if (line.trimmed().isEmpty()) continue;
        QJsonParseError parse;
        const auto document = QJsonDocument::fromJson(line, &parse);
        if (parse.error != QJsonParseError::NoError || !document.isObject()) {
            fail(tr("Invalid engine response: %1").arg(parse.errorString())); return;
        }
        event(document.object());
        if (stopping_) return;
    }
    if (output_.size() > MaxLine) fail(tr("Engine event was not terminated within the size limit."));
}

void EngineSession::event(const QJsonObject& object) {
    const auto type = object.value("type").toString();
    if (type == "ready") {
        if (ready_ || object.value("protocol").toString() != "silicon-chat-v1" ||
            object.value("max_context").toInt() != model_.maxContext ||
            !object.value("streaming").isBool() || object.value("streaming").toBool() != model_.streaming) {
            fail(tr("Engine capabilities do not match the catalog.")); return;
        }
        ready_ = true; sendPending(); return;
    }
    if (!busy() || object.value("request_id").toString() != requestId_) {
        fail(tr("Engine event belongs to an unexpected request.")); return;
    }
    if (type == "status") {
        const auto stage = object.value("stage").toString();
        if (stage == "prefill") emit statusChanged(tr("Preparing prompt…"));
        else if (stage == "decode") emit statusChanged(tr("Generating…"));
        else { fail(tr("Unknown engine stage.")); }
    } else if (type == "delta") {
        const auto text = object.value("text");
        const auto tokens = object.value("generated_tokens");
        const double count = tokens.toDouble(-1);
        if (!model_.streaming || !text.isString() || !tokens.isDouble() || count != std::floor(count) ||
            count < generated_ || count > requestedLimit_) {
            fail(tr("Invalid streaming event.")); return;
        }
        response_.append(text.toString()); generated_ = static_cast<int>(count); streamed_ = true;
        if (response_.toUtf8().size() > MaxResponse) { fail(tr("Response exceeds the memory limit.")); return; }
        if (firstTokenMs_ < 0 && !text.toString().isEmpty()) firstTokenMs_ = monotonicMilliseconds() - requestStarted_;
        emit responseChanged(response_);
    } else if (type == "completed") {
        const auto text = object.value("text");
        auto stats = object.value("statistics").toObject();
        const double count = stats.value("generated_tokens").toDouble(-1);
        const auto termination = object.value("termination").toString();
        if (!text.isString() || text.toString().toUtf8().size() > MaxResponse ||
            (streamed_ && text.toString() != response_) || count < generated_ ||
            count > requestedLimit_ || count < 0 || count != std::floor(count) ||
            (termination != "eos" && termination != "length")) {
            fail(tr("Engine completion does not match the request.")); return;
        }
        for (const auto key : {"decode_seconds", "prefill_seconds", "load_seconds", "working_set_bytes"}) {
            if (!validMetric(stats, key)) { fail(tr("Invalid engine statistics.")); return; }
        }
        stats.insert("total_seconds", (monotonicMilliseconds() - requestStarted_) / 1000.0);
        stats.remove("first_token_seconds");
        if (model_.streaming && firstTokenMs_ >= 0) stats.insert("first_token_seconds", firstTokenMs_ / 1000.0);
        stats.insert("termination", termination);
        sampleProcessMemory(stats, process_.processId());
        response_ = text.toString(); deadline_.stop(); requestId_.clear(); pending_ = {};
        emit completed(response_, stats, false);
    } else if (type == "error") {
        fail(object.value("message").toString(tr("Engine error.")));
    } else {
        fail(tr("Unsupported engine event type."));
    }
}

void EngineSession::stopProcess() {
    stopping_ = true;
#ifdef Q_OS_WIN
    if (processJob_) { CloseHandle(static_cast<HANDLE>(processJob_)); processJob_ = nullptr; }
#endif
    if (process_.state() != QProcess::NotRunning) {
        process_.kill();
        process_.waitForFinished(3000);
    }
    ready_ = false; output_.clear(); errors_.clear(); stopping_ = false;
}

void EngineSession::cancel() {
    if (!busy()) return;
    QJsonObject stats{{"total_seconds", (monotonicMilliseconds() - requestStarted_) / 1000.0},
                      {"termination", "interrupted"}};
    sampleProcessMemory(stats, process_.processId());
    deadline_.stop(); stopProcess();
    if (streamed_) stats.insert("generated_tokens", generated_);
    requestId_.clear(); pending_ = {};
    emit statusChanged(tr("Generation stopped. The engine will reload on the next message."));
    emit completed(response_, stats, true);
}

void EngineSession::fail(const QString& error) {
    const bool hadRequest = busy();
    deadline_.stop(); stopProcess();
    requestId_.clear(); pending_ = {};
    if (hadRequest && !response_.isEmpty()) emit completed(response_, QJsonObject{{"termination", "error"}}, true);
    emit failed(error);
}

void EngineSession::shutdown() {
    deadline_.stop(); stopProcess(); requestId_.clear(); pending_ = {};
}
} // namespace silicon
