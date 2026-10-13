#pragma once

#include "engine.h"
#include "mainwindow.h"
#include "store.h"

#include <QElapsedTimer>
#include <QFutureWatcher>

namespace silicon {

struct CatalogResult {
    QList<ModelSpec> models;
    QString error;
};

class AppController final : public QObject {
public:
    AppController(MainWindow& window, QString dataDirectory, QObject* parent = nullptr);
    ~AppController() override;
    void loadCatalog(const QString& file);

private:
    Conversation* conversation();
    const ModelSpec* model(const QString& id) const;
    void refresh();
    bool save(const Conversation& chat);
    void create(const QString& modelId);
    void select(const QString& id);
    void send(const QString& text);

    MainWindow& window_;
    ConversationStore store_;
    EngineSession engine_;
    QList<Conversation> conversations_;
    QList<ModelSpec> models_;
    QString selected_;
    QString loadingCatalog_;
    QFutureWatcher<CatalogResult> catalogWatcher_;
    QElapsedTimer visibleResponseTimer_;
    qint64 firstVisibleMs_ = -1;
};

} // namespace silicon
