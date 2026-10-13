#pragma once

#include "types.h"
#include <QObject>
#include <QProcess>
#include <QTimer>

namespace silicon {
class EngineSession : public QObject {
    Q_OBJECT
public:
    explicit EngineSession(QObject* parent = nullptr);
    ~EngineSession() override;
    bool generate(const ModelSpec& model, const Conversation& conversation, int maxNewTokens);
    void cancel();
    void shutdown();
    bool busy() const { return !requestId_.isEmpty(); }
    qint64 processId() const { return process_.processId(); }
signals:
    void statusChanged(const QString& status);
    void responseChanged(const QString& text);
    void completed(const QString& text, const QJsonObject& statistics, bool interrupted);
    void failed(const QString& error);
private:
    void readOutput();
    void readErrors();
    void event(const QJsonObject& object);
    void sendPending();
    void fail(const QString& error);
    void stopProcess();
    QProcess process_;
    QTimer deadline_;
    QByteArray output_;
    QByteArray errors_;
    QJsonObject pending_;
    ModelSpec model_;
    QString requestId_;
    QString response_;
    bool ready_ = false;
    bool stopping_ = false;
    bool streamed_ = false;
    int generated_ = 0;
    int requestedLimit_ = 0;
    qint64 requestStarted_ = 0;
    qint64 firstTokenMs_ = -1;
#ifdef Q_OS_WIN
    void* processJob_ = nullptr;
#endif
};
} // namespace silicon
