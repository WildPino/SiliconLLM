#include "mainwindow.h"

#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonValue>
#include <QKeyEvent>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QTextBrowser>
#include <QTextDocument>
#include <QVBoxLayout>
#include <QRegularExpression>
#include <QFrame>
#include <QClipboard>
#include <QDesktopServices>
#include <QUrl>
#include <QColor>
#include <QListView>
#include <QAbstractItemView>
#include <QLineEdit>
#include <QStandardItemModel>
#include <QtMath>
#include <functional>

namespace silicon {
namespace {

class SafeMarkdownBrowser final : public QTextBrowser {
public:
    explicit SafeMarkdownBrowser(QWidget *parent = nullptr) : QTextBrowser(parent) {}

protected:
    QVariant loadResource(int, const QUrl &) override { return {}; }
};

class SafeMarkdownDocument final : public QTextDocument {
public:
    explicit SafeMarkdownDocument(QObject *parent = nullptr) : QTextDocument(parent) {}

protected:
    QVariant loadResource(int, const QUrl &) override { return {}; }
};

QString metric(const QJsonObject &stats, const QString &key, const QString &label,
               const QString &suffix = QString()) {
    const auto value = stats.value(key);
    if (!value.isDouble()) return {};
    const double number = value.toDouble();
    QString rendered;
    if (key == QStringLiteral("working_set_bytes")) {
        rendered = QString::number(number / (1024.0 * 1024.0), 'f', 0) + QStringLiteral(" MiB");
    } else if (key == QStringLiteral("generated_tokens")) {
        rendered = QString::number(qRound64(number));
    } else {
        rendered = QString::number(number, 'f', 2) + suffix;
    }
    return label + QStringLiteral(" ") + rendered;
}

QString statisticsText(const QJsonObject &stats) {
    QStringList items;
    const auto add = [&items](const QString &s) { if (!s.isEmpty()) items.append(s); };
    add(metric(stats, QStringLiteral("generated_tokens"), QStringLiteral("Generated tokens")));
    add(metric(stats, QStringLiteral("total_seconds"), QStringLiteral("Total"), QStringLiteral(" s")));
    const auto tokens = stats.value(QStringLiteral("generated_tokens"));
    const auto decode = stats.value(QStringLiteral("decode_seconds"));
    if (tokens.isDouble() && decode.isDouble() && decode.toDouble() > 0.0)
        items.append(QStringLiteral("Decode ") + QString::number(tokens.toDouble() / decode.toDouble(), 'f', 1) + QStringLiteral(" tok/s"));
    add(metric(stats, QStringLiteral("working_set_bytes"), QStringLiteral("Process memory")));
    add(metric(stats, QStringLiteral("first_token_seconds"), QStringLiteral("First text in UI"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("prefill_seconds"), QStringLiteral("Prompt"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("decode_seconds"), QStringLiteral("Decode time"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("load_seconds"), QStringLiteral("Load"), QStringLiteral(" s")));
    return items.join(QStringLiteral("  ·  "));
}

QTextBrowser *makeMarkdownView(const QString &markdown, QWidget *parent) {
    auto *view = new SafeMarkdownBrowser(parent);
    auto *document = new SafeMarkdownDocument(view);
    view->setDocument(document);
    view->setObjectName(QStringLiteral("messageMarkdown"));
    view->setFrameShape(QFrame::NoFrame);
    view->setOpenExternalLinks(false);
    view->setOpenLinks(false);
    view->setReadOnly(true);
    view->setContextMenuPolicy(Qt::DefaultContextMenu);
    QObject::connect(view, &QTextBrowser::anchorClicked, view, [](const QUrl &url) {
        if (url.scheme() == QStringLiteral("https") || url.scheme() == QStringLiteral("http"))
            QDesktopServices::openUrl(url);
    });
    view->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    view->setStyleSheet(QStringLiteral("QTextBrowser { background: transparent; border: 0; padding: 0; color: #292923; selection-background-color: #d9e5df; }"));
    document->setMarkdown(markdown, QTextDocument::MarkdownNoHTML);
    view->setMinimumHeight(28);
    view->setMaximumHeight(1000);
    return view;
}

class DraftEditor final : public QPlainTextEdit {
public:
    explicit DraftEditor(QWidget *parent = nullptr) : QPlainTextEdit(parent) {}
    std::function<void()> send;
protected:
    void keyPressEvent(QKeyEvent *event) override {
        if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
            if (!(event->modifiers() & Qt::ShiftModifier)) {
                if (send) send();
                event->accept();
                return;
            }
        }
        QPlainTextEdit::keyPressEvent(event);
    }
};

} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    buildUi();
    draftTimer_.setSingleShot(true);
    draftTimer_.setInterval(75);
    connect(&draftTimer_, &QTimer::timeout, this, [this] { updateDraftView(false); });
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("SiliconLLM · Local chat"));
    resize(1180, 780);
    setMinimumSize(760, 560);
    setStyleSheet(QStringLiteral(
        "QMainWindow, QWidget { background: #f7f6f2; color: #292923; font-family: 'Segoe UI'; font-size: 10pt; }"
        "QListWidget, QPlainTextEdit, QComboBox { background: #fffefa; border: 1px solid #deddd5; border-radius: 8px; padding: 8px; }"
        "QListWidget::item { padding: 9px 8px; border-radius: 6px; } QListWidget::item:selected { background: #e6ebe6; color: #252b27; }"
        "QPushButton { background: #eeede7; border: 1px solid #deddd5; border-radius: 7px; padding: 7px 11px; }"
        "QPushButton:hover { background: #e5e4dc; } QPushButton:disabled { color: #a7a69f; }"
        "QPushButton#sendButton { background: #34463c; border-color: #34463c; color: white; font-weight: 600; }"
        "QPushButton#stopButton { background: #f4e9e5; color: #7a3023; }"
        "QPushButton#sendButton:disabled, QPushButton#stopButton:disabled { background: #eeede7; border-color: #deddd5; color: #a7a69f; }"
        "QLabel#statusLabel { color: #66675f; } QLabel#emptyState { color: #66675f; }"));

    auto *root = new QWidget(this);
    auto *rootLayout = new QHBoxLayout(root);
    rootLayout->setContentsMargins(16, 14, 16, 14);
    rootLayout->setSpacing(14);

    auto *sidebar = new QWidget(root);
    sidebar->setFixedWidth(258);
    auto *sideLayout = new QVBoxLayout(sidebar);
    sideLayout->setContentsMargins(0, 0, 0, 0);
    sideLayout->setSpacing(9);
    auto *brand = new QLabel(QStringLiteral("SiliconLLM"), sidebar);
    QFont brandFont = brand->font(); brandFont.setPointSize(16); brandFont.setWeight(QFont::DemiBold); brand->setFont(brandFont);
    sideLayout->addWidget(brand);
    auto *newChat = new QPushButton(QStringLiteral("＋  New chat"), sidebar);
    connect(newChat, &QPushButton::clicked, this, [this] {
        if (busy_) return;
        const QString id = modelSelector_->currentData().toString();
        if (!id.isEmpty()) emit newConversationRequested(id);
    });
    sideLayout->addWidget(newChat);
    conversationList_ = new QListWidget(sidebar);
    conversationList_->setObjectName(QStringLiteral("conversationList"));
    conversationList_->setSelectionMode(QAbstractItemView::SingleSelection);
    sideLayout->addWidget(conversationList_, 1);
    connect(conversationList_, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *item) {
        if (!busy_ && item) emit conversationSelected(item->data(Qt::UserRole).toString());
    });
    auto *chatActions = new QHBoxLayout;
    auto *rename = new QPushButton(QStringLiteral("Rename"), sidebar);
    auto *remove = new QPushButton(QStringLiteral("Delete"), sidebar);
    chatActions->addWidget(rename); chatActions->addWidget(remove);
    sideLayout->addLayout(chatActions);
    connect(rename, &QPushButton::clicked, this, [this] {
        auto *item = conversationList_->currentItem();
        if (!item || busy_) return;
        bool accepted = false;
        const QString title = QInputDialog::getText(this, QStringLiteral("Rename conversation"),
            QStringLiteral("Conversation name"), QLineEdit::Normal, item->text(), &accepted).trimmed();
        if (accepted && !title.isEmpty()) emit renameConversationRequested(item->data(Qt::UserRole).toString(), title);
    });
    connect(remove, &QPushButton::clicked, this, [this] {
        auto *item = conversationList_->currentItem();
        if (item && !busy_) emit deleteConversationRequested(item->data(Qt::UserRole).toString());
    });

    auto *main = new QWidget(root);
    auto *mainLayout = new QVBoxLayout(main);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(10);
    auto *top = new QHBoxLayout;
    auto *modelLabel = new QLabel(QStringLiteral("Model"), main);
    modelSelector_ = new QComboBox(main);
    modelSelector_->setObjectName(QStringLiteral("modelSelector"));
    modelSelector_->setMinimumWidth(280);
    modelSelector_->setEnabled(false);
    modelDescription_ = new QLabel(main);
    modelDescription_->setStyleSheet(QStringLiteral("color:#73736b;"));
    catalogButton_ = new QPushButton(QStringLiteral("Import model catalog…"), main);
    catalogButton_->setObjectName(QStringLiteral("catalogButton"));
    top->addWidget(modelLabel); top->addWidget(modelSelector_); top->addWidget(modelDescription_, 1); top->addWidget(catalogButton_);
    mainLayout->addLayout(top);
    connect(catalogButton_, &QPushButton::clicked, this, &MainWindow::catalogRequested);
    connect(modelSelector_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        updateModelDescription();
        updateSendAvailability();
        if (index >= 0 && !busy_) emit modelChangeRequested(modelSelector_->itemData(index).toString());
    });

    transcriptScroll_ = new QScrollArea(main);
    transcriptScroll_->setWidgetResizable(true);
    transcriptScroll_->setFrameShape(QFrame::NoFrame);
    transcriptScroll_->setObjectName(QStringLiteral("transcriptScroll"));
    transcriptContent_ = new QWidget(transcriptScroll_);
    transcriptContent_->setObjectName(QStringLiteral("chatTranscript"));
    transcriptLayout_ = new QVBoxLayout(transcriptContent_);
    transcriptLayout_->setContentsMargins(22, 20, 22, 20);
    transcriptLayout_->setSpacing(16);
    transcriptLayout_->addStretch(1);
    transcriptScroll_->setWidget(transcriptContent_);
    emptyState_ = new QLabel(QStringLiteral("Choose an approved model, then start a conversation.\nUse “Import model catalog…” to load the validated local catalog."), transcriptContent_);
    emptyState_->setObjectName(QStringLiteral("emptyState"));
    emptyState_->setAlignment(Qt::AlignCenter);
    emptyState_->setWordWrap(true);
    transcriptLayout_->insertWidget(0, emptyState_);
    mainLayout->addWidget(transcriptScroll_, 1);

    statusLabel_ = new QLabel(QStringLiteral("Ready"), main);
    statusLabel_->setObjectName(QStringLiteral("statusLabel"));
    mainLayout->addWidget(statusLabel_);
    auto *composer = new QHBoxLayout;
    auto *editor = new DraftEditor(main);
    messageInput_ = editor;
    messageInput_->setObjectName(QStringLiteral("messageInput"));
    messageInput_->setPlaceholderText(QStringLiteral("Message your model…  (Enter to send, Shift+Enter for a new line)"));
    messageInput_->setMinimumHeight(76);
    messageInput_->setMaximumHeight(150);
    editor->send = [this] { submitDraft(); };
    auto *buttons = new QVBoxLayout;
    sendButton_ = new QPushButton(QStringLiteral("Send"), main);
    sendButton_->setObjectName(QStringLiteral("sendButton"));
    sendButton_->setEnabled(false);
    stopButton_ = new QPushButton(QStringLiteral("Stop"), main);
    stopButton_->setObjectName(QStringLiteral("stopButton"));
    stopButton_->setEnabled(false);
    buttons->addWidget(sendButton_); buttons->addWidget(stopButton_); buttons->addStretch();
    composer->addWidget(messageInput_, 1); composer->addLayout(buttons);
    mainLayout->addLayout(composer);
    connect(sendButton_, &QPushButton::clicked, this, [this] { submitDraft(); });
    connect(stopButton_, &QPushButton::clicked, this, &MainWindow::stopRequested);
    rootLayout->addWidget(sidebar);
    rootLayout->addWidget(main, 1);
    setCentralWidget(root);
}

void MainWindow::setModels(const QList<ModelSpec> &models) {
    const QString previous = hasConversation_ ? currentConversation_.modelId : modelSelector_->currentData().toString();
    models_ = models;
    modelSelector_->blockSignals(true);
    modelSelector_->clear();
    bool hasAvailable = false;
    for (const auto &model : models_) {
        const bool usable = model.approved && model.available;
        QString label = model.name;
        if (!model.revision.isEmpty()) label += QStringLiteral(" · ") + model.revision;
        if (model.maxContext > 0) label += QStringLiteral(" · ") + QString::number(model.maxContext) + QStringLiteral(" ctx");
        if (!usable) {
            const QString reason = !model.unavailableReason.isEmpty() ? model.unavailableReason :
                (!model.approved ? QStringLiteral("Not approved") : QStringLiteral("Unavailable"));
            label += QStringLiteral(" — ") + reason;
        }
        modelSelector_->addItem(label, model.id);
        const int row = modelSelector_->count() - 1;
        modelSelector_->setItemData(row, usable, Qt::UserRole + 1);
        modelSelector_->setItemData(row, model.unavailableReason, Qt::ToolTipRole);
        modelSelector_->setItemData(row, QColor(QStringLiteral("#9a9992")), Qt::ForegroundRole);
        if (auto *standardModel = qobject_cast<QStandardItemModel *>(modelSelector_->model()))
            standardModel->item(row)->setEnabled(usable);
        hasAvailable |= usable;
    }
    int selected = modelSelector_->findData(previous);
    if (selected >= 0 && !modelSelector_->itemData(selected, Qt::UserRole + 1).toBool()) selected = -1;
    if (selected < 0) {
        selected = -1;
        for (int i = 0; i < modelSelector_->count(); ++i)
            if (modelSelector_->itemData(i, Qt::UserRole + 1).toBool()) { selected = i; break; }
    }
    modelSelector_->setCurrentIndex(selected);
    modelSelector_->setEnabled(hasAvailable && !busy_);
    modelSelector_->setPlaceholderText(hasAvailable ? QStringLiteral("Select a model") : QStringLiteral("Import an approved model catalog"));
    modelSelector_->blockSignals(false);
    if (!hasAvailable) {
        QString explanation = models_.isEmpty()
            ? QStringLiteral("No model catalog is loaded. Import a catalog containing an approved, available model.")
            : QStringLiteral("No approved model is available.");
        for (const auto &model : models_) {
            if (!model.unavailableReason.isEmpty()) {
                explanation += QStringLiteral("\n") + model.name + QStringLiteral(": ") + model.unavailableReason;
                break;
            }
        }
        emptyState_->setText(explanation);
    } else {
        emptyState_->setText(QStringLiteral("Choose an approved model, then start a conversation.\nUse “Import model catalog…” to load the validated local catalog."));
    }
    updateModelDescription();
    updateSendAvailability();
}

void MainWindow::setConversations(const QList<Conversation> &conversations) {
    conversations_ = conversations;
    const QString currentId = hasConversation_ ? currentConversation_.id : QString();
    conversationList_->blockSignals(true);
    conversationList_->clear();
    int selected = -1;
    for (const auto &conversation : conversations_) {
        auto *item = new QListWidgetItem(conversation.title.isEmpty() ? QStringLiteral("New conversation") : conversation.title, conversationList_);
        item->setData(Qt::UserRole, conversation.id);
        if (conversation.id == currentId) selected = conversationList_->count() - 1;
    }
    if (selected >= 0) conversationList_->setCurrentRow(selected);
    conversationList_->setEnabled(!busy_);
    conversationList_->blockSignals(false);
}

void MainWindow::showConversation(const Conversation &conversation) {
    currentConversation_ = conversation;
    hasConversation_ = !conversation.id.isEmpty();
    if (!hasConversation_) currentConversation_ = Conversation{};
    draft_.clear();
    const int modelIndex = hasConversation_ ? modelSelector_->findData(conversation.modelId) : modelSelector_->currentIndex();
    modelSelector_->blockSignals(true);
    modelSelector_->setCurrentIndex(modelIndex);
    modelSelector_->blockSignals(false);
    updateModelDescription();
    updateSendAvailability();
    renderConversation();
    for (int i = 0; i < conversationList_->count(); ++i) {
        auto *item = conversationList_->item(i);
        if (item->data(Qt::UserRole).toString() == conversation.id) {
            conversationList_->blockSignals(true); conversationList_->setCurrentItem(item); conversationList_->blockSignals(false); break;
        }
    }
    emptyState_->setVisible(!hasConversation_ || (conversation.messages.isEmpty() && draft_.isEmpty()));
    scrollToBottom();
}

void MainWindow::clearTranscript() {
    while (transcriptLayout_->count() > 1) {
        QLayoutItem *item = transcriptLayout_->takeAt(0);
        if (item->widget() && item->widget() != emptyState_) item->widget()->deleteLater();
        delete item;
    }
    transcriptLayout_->insertWidget(0, emptyState_);
    emptyState_->setVisible(!hasConversation_ || currentConversation_.messages.isEmpty());
}

void MainWindow::renderConversation() {
    clearTranscript();
    draftBrowser_ = nullptr;
    if (!hasConversation_) return;
    for (const auto &message : currentConversation_.messages) appendMessage(message);
    if (!draft_.isEmpty()) {
        Message draftMessage; draftMessage.role = QStringLiteral("assistant"); draftMessage.content = draft_;
        appendMessage(draftMessage, true);
    }
    emptyState_->setVisible(currentConversation_.messages.isEmpty() && draft_.isEmpty());
    scrollToBottom();
}

void MainWindow::appendMessage(const Message &message, bool draft) {
    auto *card = new QWidget(transcriptContent_);
    card->setObjectName(QStringLiteral("messageCard"));
    card->setStyleSheet(QStringLiteral("QWidget#messageCard { background:#fffefa; border:1px solid #e6e4dc; border-radius:10px; }"));
    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(14, 10, 14, 10);
    layout->setSpacing(5);
    auto *header = new QHBoxLayout;
    auto *role = new QLabel(message.role.compare(QStringLiteral("assistant"), Qt::CaseInsensitive) == 0 ? QStringLiteral("SiliconLLM") : QStringLiteral("You"), card);
    QFont f = role->font(); f.setWeight(QFont::DemiBold); role->setFont(f);
    header->addWidget(role); header->addStretch();
    auto *copy = new QPushButton(QStringLiteral("Copy"), card);
    copy->setMaximumWidth(72);
    connect(copy, &QPushButton::clicked, this, [text = message.content] { QApplication::clipboard()->setText(text); });
    header->addWidget(copy);
    layout->addLayout(header);
    auto *body = makeMarkdownView(message.content, card);
    if (draft) {
        body->setHtml(QString());
        body->setPlainText(message.content);
        draftBrowser_ = body;
    }
    body->setMinimumHeight(qMin(450, qMax(38, qCeil(body->document()->size().height() + 10))));
    layout->addWidget(body);
    if (!draft && message.role.compare(QStringLiteral("assistant"), Qt::CaseInsensitive) == 0) {
        const QString stats = statisticsText(message.statistics);
        if (!stats.isEmpty()) {
            auto *details = new QLabel(stats, card);
            details->setObjectName(QStringLiteral("messageStatistics"));
            details->setStyleSheet(QStringLiteral("color:#77776f; font-size:9pt; border-top:1px solid #eeede7; padding-top:6px;"));
            details->setWordWrap(true);
            layout->addWidget(details);
        }
    }
    transcriptLayout_->insertWidget(transcriptLayout_->count() - 1, card);

    static const QRegularExpression codePattern(QStringLiteral("```[^\\n]*\\n([\\s\\S]*?)```"));
    auto matches = codePattern.globalMatch(message.content);
    int index = 0;
    while (matches.hasNext()) {
        const QString code = matches.next().captured(1);
        auto *copyCode = new QPushButton(QStringLiteral("Copy code %1").arg(++index), card);
        copyCode->setMaximumWidth(120);
        connect(copyCode, &QPushButton::clicked, this, [code] { QApplication::clipboard()->setText(code); });
        layout->addWidget(copyCode, 0, Qt::AlignRight);
    }
}

void MainWindow::setBusy(bool busy, const QString &status) {
    if (busy && !busy_) firstContentDisplayedForRequest_ = false;
    busy_ = busy;
    statusLabel_->setText(status);
    updateSendAvailability();
    stopButton_->setEnabled(busy);
    catalogButton_->setEnabled(!busy);
    messageInput_->setEnabled(!busy);
    conversationList_->setEnabled(!busy);
    bool hasAvailableModel = false;
    for (int i = 0; i < modelSelector_->count(); ++i)
        hasAvailableModel |= modelSelector_->itemData(i, Qt::UserRole + 1).toBool();
    modelSelector_->setEnabled(!busy && hasAvailableModel);
    if (!busy) {
        draftTimer_.stop();
        updateDraftView(true);
    }
}

void MainWindow::updateSendAvailability() {
    if (!sendButton_) return;
    const int index = modelSelector_->currentIndex();
    bool canSend = index >= 0 && modelSelector_->itemData(index, Qt::UserRole + 1).toBool();
    if (hasConversation_)
        canSend = canSend && modelSelector_->itemData(index).toString() == currentConversation_.modelId;
    sendButton_->setEnabled(!busy_ && canSend);
}

void MainWindow::setDraftResponse(const QString &cumulativeText) {
    draft_ = cumulativeText;
    if (!draftBrowser_ && !draft_.isEmpty()) renderConversation();
    if (busy_ && !draft_.isEmpty() && !firstContentDisplayedForRequest_) {
        draftTimer_.stop();
        updateDraftView(false);
    } else if (busy_) draftTimer_.start();
    else updateDraftView(true);
}

void MainWindow::updateDraftView(bool finalize) {
    if (!draftBrowser_) return;
    if (!finalize) {
        draftBrowser_->setPlainText(draft_);
        draftBrowser_->setMinimumHeight(qMin(450, qMax(38, qCeil(draftBrowser_->document()->size().height() + 10))));
        if (busy_ && !draft_.isEmpty() && !firstContentDisplayedForRequest_) {
            firstContentDisplayedForRequest_ = true;
            emit firstContentDisplayed();
        }
        scrollToBottom();
        return;
    }
    draftBrowser_->document()->setMarkdown(draft_, QTextDocument::MarkdownNoHTML);
    draftBrowser_->setMinimumHeight(qMin(450, qMax(38, qCeil(draftBrowser_->document()->size().height() + 10))));
    scrollToBottom();
}

void MainWindow::showError(const QString &message) {
    statusLabel_->setText(QStringLiteral("Error: ") + message);
    statusLabel_->setStyleSheet(QStringLiteral("color:#8b3528;"));
}

void MainWindow::updateModelDescription() {
    const QString id = modelSelector_->currentData().toString();
    for (const auto &model : models_) {
        if (model.id != id) continue;
        QString text;
        if (!model.approved || !model.available) text = !model.unavailableReason.isEmpty() ? model.unavailableReason : QStringLiteral("Model unavailable");
        else text = model.streaming ? QStringLiteral("Progressive output") : QStringLiteral("Response appears when generation completes");
        modelDescription_->setText(text);
        modelDescription_->setToolTip(model.qualification.value(QStringLiteral("reason")).toString());
        return;
    }
    modelDescription_->setText(QStringLiteral("No approved model available"));
}

void MainWindow::submitDraft() {
    if (busy_ || !modelSelector_->currentData().isValid()) return;
    const QString text = messageInput_->toPlainText().trimmed();
    if (text.isEmpty()) return;
    messageInput_->clear();
    emit sendRequested(text);
}

bool MainWindow::eventFilter(QObject *watched, QEvent *event) {
    return QMainWindow::eventFilter(watched, event);
}

void MainWindow::scrollToBottom() {
    QMetaObject::invokeMethod(this, [this] { transcriptScroll_->verticalScrollBar()->setValue(transcriptScroll_->verticalScrollBar()->maximum()); }, Qt::QueuedConnection);
}

} // namespace silicon
