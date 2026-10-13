#include "controller.h"

#include <QFileDialog>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSettings>
#include <QTimer>
#include <QUuid>
#include <QtConcurrent/QtConcurrentRun>
#include <algorithm>
#include <utility>

namespace silicon {

AppController::AppController(MainWindow& window, QString dataDirectory, QObject* parent)
    : QObject(parent), window_(window), store_(std::move(dataDirectory)), engine_(this) {
    QString error;
    conversations_ = store_.load(&error);
    refresh();
    if (!conversations_.isEmpty()) select(conversations_.first().id);
    if (!error.isEmpty()) QTimer::singleShot(0, this, [this, error] { window_.showError(error); });
    connect(&window_, &MainWindow::catalogRequested, this, [this] {
        if (engine_.busy() || catalogWatcher_.isRunning()) return;
        const auto file = QFileDialog::getOpenFileName(&window_, tr("Model catalog"), {}, tr("JSON catalogs (*.json)"));
        if (!file.isEmpty()) loadCatalog(file);
    });
    connect(&window_, &MainWindow::newConversationRequested, this, &AppController::create);
    connect(&window_, &MainWindow::modelChangeRequested, this, [this](const QString& id) {
        auto* current = conversation();
        if (!current || current->modelId != id) create(id);
    });
    connect(&window_, &MainWindow::conversationSelected, this, &AppController::select);
    connect(&window_, &MainWindow::renameConversationRequested, this, [this](const QString& id, const QString& title) {
        if (engine_.busy()) return;
        for (auto& chat : conversations_) if (chat.id == id) {
            chat.title = title.trimmed().left(200); if (chat.title.isEmpty()) return;
            chat.updatedAt = QDateTime::currentDateTimeUtc(); save(chat); refresh(); select(id); return;
        }
    });
    connect(&window_, &MainWindow::deleteConversationRequested, this, [this](const QString& id) {
        if (engine_.busy()) return;
        QString error;
        if (!store_.remove(id, &error)) { window_.showError(error); return; }
        conversations_.removeIf([&id](const auto& chat) { return chat.id == id; });
        if (selected_ == id) { selected_.clear(); window_.showConversation({}); }
        refresh();
        if (selected_.isEmpty() && !conversations_.isEmpty()) select(conversations_.first().id);
    });
    connect(&window_, &MainWindow::sendRequested, this, &AppController::send);
    connect(&window_, &MainWindow::stopRequested, &engine_, &EngineSession::cancel);
    connect(&window_, &MainWindow::firstContentDisplayed, this, [this] {
        if (visibleResponseTimer_.isValid() && firstVisibleMs_ < 0) firstVisibleMs_ = visibleResponseTimer_.elapsed();
    });
    connect(&engine_, &EngineSession::statusChanged, this, [this](const QString& status) {
        window_.setBusy(engine_.busy(), status);
    });
    connect(&engine_, &EngineSession::responseChanged, &window_, &MainWindow::setDraftResponse);
    connect(&engine_, &EngineSession::completed, this, [this](const QString& text, const QJsonObject& statistics, bool interrupted) {
        if (auto* chat = conversation()) {
            QJsonObject stats = statistics;
            if (firstVisibleMs_ >= 0)
                stats.insert(QStringLiteral("first_token_seconds"), firstVisibleMs_ / 1000.0);
            else
                stats.remove(QStringLiteral("first_token_seconds"));
            firstVisibleMs_ = -1;
            visibleResponseTimer_.invalidate();
            chat->messages.append(Message{"assistant", text, interrupted, stats});
            chat->updatedAt = QDateTime::currentDateTimeUtc(); save(*chat);
            window_.setBusy(false, interrupted ? tr("Stopped. The engine will reload on the next message.") : tr("Ready"));
            refresh(); window_.showConversation(*chat);
        }
    });
    connect(&engine_, &EngineSession::failed, this, [this](const QString& error) {
        firstVisibleMs_ = -1;
        visibleResponseTimer_.invalidate();
        window_.setBusy(engine_.busy(), tr("Engine error")); window_.showError(error);
    });
    connect(&catalogWatcher_, &QFutureWatcher<CatalogResult>::finished, this, [this] {
        const auto result = catalogWatcher_.result();
        if (!result.error.isEmpty()) { window_.setBusy(false, tr("Invalid catalog")); window_.showError(result.error); return; }
        engine_.shutdown(); models_ = result.models; window_.setModels(models_);
        window_.setBusy(false, models_.isEmpty() ? tr("No approved CPU runtime installed") : tr("Catalog verified"));
        QSettings().setValue("catalog", loadingCatalog_);
        if (auto* chat = conversation()) window_.showConversation(*chat);
    });
}

AppController::~AppController() { engine_.shutdown(); catalogWatcher_.waitForFinished(); }

void AppController::loadCatalog(const QString& file) {
    if (engine_.busy() || catalogWatcher_.isRunning()) return;
    loadingCatalog_ = QFileInfo(file).absoluteFilePath();
    window_.setBusy(true, tr("Verifying catalog and artifacts…"));
    catalogWatcher_.setFuture(QtConcurrent::run([file] {
        CatalogResult result; result.models = ModelCatalog::load(file, &result.error); return result;
    }));
}

Conversation* AppController::conversation() {
    for (auto& chat : conversations_) if (chat.id == selected_) return &chat;
    return nullptr;
}

const ModelSpec* AppController::model(const QString& id) const {
    for (const auto& spec : models_) if (spec.id == id && spec.available && spec.approved) return &spec;
    return nullptr;
}

void AppController::refresh() { window_.setConversations(conversations_); }

bool AppController::save(const Conversation& chat) {
    QString error; if (store_.save(chat, &error)) return true;
    window_.showError(tr("Cannot save conversation: %1").arg(error)); return false;
}

void AppController::create(const QString& modelId) {
    if (engine_.busy() || catalogWatcher_.isRunning() || !model(modelId)) return;
    Conversation chat{QUuid::createUuid().toString(QUuid::WithoutBraces), tr("New conversation"), modelId, {}, QDateTime::currentDateTimeUtc()};
    if (!save(chat)) return;
    conversations_.prepend(chat); selected_ = chat.id; refresh(); window_.showConversation(chat);
}

void AppController::select(const QString& id) {
    if (engine_.busy() || catalogWatcher_.isRunning()) return;
    selected_ = id;
    if (auto* chat = conversation()) window_.showConversation(*chat);
}

void AppController::send(const QString& text) {
    if (engine_.busy() || catalogWatcher_.isRunning() || text.trimmed().isEmpty()) return;
    auto* chat = conversation();
    if (!chat) { window_.showError(tr("Create a conversation with an approved model.")); return; }
    const auto* spec = model(chat->modelId);
    if (!spec) { window_.showError(tr("The model for this conversation is unavailable in the catalog.")); return; }
    if (text.toUtf8().size() > 1024 * 1024) { window_.showError(tr("Message exceeds the 1 MiB limit.")); return; }
    chat->messages.append(Message{"user", text, false, {}});
    if (chat->messages.size() == 1) chat->title = text.simplified().left(60);
    chat->updatedAt = QDateTime::currentDateTimeUtc();
    if (!save(*chat)) { chat->messages.removeLast(); return; }
    refresh(); window_.showConversation(*chat);
    firstVisibleMs_ = -1;
    visibleResponseTimer_.start();
    window_.setBusy(true, tr("Starting…"));
    if (!engine_.generate(*spec, *chat, std::min(256, spec->maxNewTokens))) {
        firstVisibleMs_ = -1;
        visibleResponseTimer_.invalidate();
        window_.setBusy(false, tr("Send failed"));
    }
}

} // namespace silicon
