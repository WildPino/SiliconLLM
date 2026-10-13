#pragma once

#include "types.h"

#include <QString>

namespace silicon {

class ConversationStore {
public:
    explicit ConversationStore(QString directory);

    QList<Conversation> load(QString *error = nullptr) const;
    bool save(const Conversation &conversation, QString *error = nullptr) const;
    bool remove(const QString &id, QString *error = nullptr) const;
    static QString defaultDirectory();

private:
    QString directory_;
};

class ModelCatalog {
public:
    static QList<ModelSpec> load(const QString &file, QString *error = nullptr);
};

} // namespace silicon
