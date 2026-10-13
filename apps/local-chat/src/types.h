#pragma once

#include <QDateTime>
#include <QJsonObject>
#include <QList>
#include <QString>
#include <QStringList>

namespace silicon {
struct Message {
    QString role;
    QString content;
    bool interrupted = false;
    QJsonObject statistics;
};

struct Conversation {
    QString id;
    QString title;
    QString modelId;
    QList<Message> messages;
    QDateTime updatedAt;
};

struct ModelSpec {
    QString id;
    QString name;
    QString revision;
    QString protocol;
    QString executable;
    QString artifact;
    QString tokenizer;
    QString python;
    QString workingDirectory;
    QJsonObject qualification;
    QStringList arguments;
    bool approved = false;
    bool available = false;
    bool streaming = false;
    QString unavailableReason;
    int maxContext = 0;
    int maxNewTokens = 0;
    int timeoutSeconds = 120;
};
} // namespace silicon
