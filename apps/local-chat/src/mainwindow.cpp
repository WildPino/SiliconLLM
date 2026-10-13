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
#include <QAbstractTextDocumentLayout>
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
#include <QSettings>
#include <QCheckBox>
#include <QFormLayout>
#include <QMenu>
#include <QSizePolicy>
#include <QAction>
#include <QStyle>
#include <functional>

#ifdef Q_OS_WIN
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dwmapi.h>
#endif

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
    add(metric(stats, QStringLiteral("generated_tokens"), QStringLiteral("Token generati")));
    add(metric(stats, QStringLiteral("total_seconds"), QStringLiteral("Durata"), QStringLiteral(" s")));
    const auto tokens = stats.value(QStringLiteral("generated_tokens"));
    const auto decode = stats.value(QStringLiteral("decode_seconds"));
    if (tokens.isDouble() && decode.isDouble() && decode.toDouble() > 0.0)
        items.append(QStringLiteral("Decode ") + QString::number(tokens.toDouble() / decode.toDouble(), 'f', 1) + QStringLiteral(" tok/s"));
    add(metric(stats, QStringLiteral("working_set_bytes"), QStringLiteral("Memoria processo")));
    add(metric(stats, QStringLiteral("first_token_seconds"), QStringLiteral("Primo testo"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("prefill_seconds"), QStringLiteral("Prompt"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("decode_seconds"), QStringLiteral("Tempo decode"), QStringLiteral(" s")));
    add(metric(stats, QStringLiteral("load_seconds"), QStringLiteral("Caricamento"), QStringLiteral(" s")));
    return items.join(QStringLiteral("  ·  "));
}

QTextBrowser *makeMarkdownView(const QString &markdown, QWidget *parent,
                               const QString &themeName, int textSize) {
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
    view->setFont(QFont(QStringLiteral("Segoe UI"), textSize));
    document->setDefaultFont(view->font());
    document->setDefaultStyleSheet(Theme::markdownStyle(themeName));
    document->setMarkdown(markdown, QTextDocument::MarkdownNoHTML);
    view->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    QObject::connect(document->documentLayout(), &QAbstractTextDocumentLayout::documentSizeChanged,
                     view, [view](const QSizeF &size) {
        const int height = qBound(38, qCeil(size.height() + 10), 450);
        if (view->height() != height) view->setFixedHeight(height);
    });
    view->setFixedHeight(qBound(38, qCeil(document->size().height() + 10), 450));
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
    QSettings settings(QStringLiteral("SiliconLLM"), QStringLiteral("LocalChat"));
    themeName_ = Theme::normalize(settings.value(QStringLiteral("ui/theme"), QStringLiteral("light")).toString());
    textSize_ = settings.value(QStringLiteral("ui/textSize"), 11).toInt();
    if (textSize_ != 11 && textSize_ != 12 && textSize_ != 14) textSize_ = 11;
    showStatistics_ = settings.value(QStringLiteral("ui/showStatistics"), true).toBool();
    staticGlass_ = settings.value(QStringLiteral("ui/staticGlass"), true).toBool();
    buildUi();
    applyAppearance(false);
    draftTimer_.setSingleShot(true);
    draftTimer_.setInterval(75);
    connect(&draftTimer_, &QTimer::timeout, this, [this] { updateDraftView(false); });
}

void MainWindow::buildUi() {
    setWindowTitle(QStringLiteral("SiliconLLM"));
    resize(1220, 820);
    setMinimumSize(820, 620);
    auto *root = new QWidget(this);
    root->setObjectName(QStringLiteral("windowRoot"));
    auto *rootLayout = new QHBoxLayout(root);
    rootLayout->setContentsMargins(18, 16, 18, 16);
    rootLayout->setSpacing(16);

    auto *sidebar = new QWidget(root);
    sidebar->setObjectName(QStringLiteral("sidebar"));
    sidebar->setFixedWidth(258);
    auto *sideLayout = new QVBoxLayout(sidebar);
    sideLayout->setContentsMargins(14, 10, 14, 12);
    sideLayout->setSpacing(12);
    auto *brand = new QLabel(QStringLiteral("SiliconLLM"), sidebar);
    brand->setObjectName(QStringLiteral("brandLabel"));
    QFont brandFont = brand->font(); brandFont.setPointSize(23); brandFont.setWeight(QFont::DemiBold); brand->setFont(brandFont);
    sideLayout->addWidget(brand);
    auto *brandCaption = new QLabel(QStringLiteral("CHAT LOCALE"), sidebar);
    brandCaption->setObjectName(QStringLiteral("brandCaption"));
    QFont captionFont = brandCaption->font(); captionFont.setPointSize(8); captionFont.setWeight(QFont::DemiBold); brandCaption->setFont(captionFont);
    sideLayout->addWidget(brandCaption);
    newChatButton_ = new QPushButton(QStringLiteral("Nuova chat"), sidebar);
    newChatButton_->setObjectName(QStringLiteral("newChatButton"));
    newChatButton_->setEnabled(false);
    connect(newChatButton_, &QPushButton::clicked, this, [this] {
        if (busy_) return;
        const QString id = modelSelector_->currentData().toString();
        if (!id.isEmpty()) emit newConversationRequested(id);
    });
    sideLayout->addWidget(newChatButton_);
    conversationList_ = new QListWidget(sidebar);
    conversationList_->setObjectName(QStringLiteral("conversationList"));
    conversationList_->setAccessibleName(QStringLiteral("Conversazioni"));
    conversationList_->setSelectionMode(QAbstractItemView::SingleSelection);
    conversationList_->setContextMenuPolicy(Qt::CustomContextMenu);
    sideLayout->addWidget(conversationList_, 1);
    connect(conversationList_, &QListWidget::currentItemChanged, this, [this](QListWidgetItem *item) {
        if (!busy_ && item) emit conversationSelected(item->data(Qt::UserRole).toString());
    });
    connect(conversationList_, &QListWidget::customContextMenuRequested,
            this, &MainWindow::showConversationMenu);

    auto *main = new QWidget(root);
    main->setObjectName(QStringLiteral("mainPane"));
    auto *mainLayout = new QVBoxLayout(main);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(12);

    auto *titleBar = new QWidget(main);
    titleBar->setObjectName(QStringLiteral("topBar"));
    auto *titleLayout = new QHBoxLayout(titleBar);
    titleLayout->setContentsMargins(16, 11, 16, 11);
    chatTitle_ = new QLabel(QStringLiteral("Nuova chat"), titleBar);
    chatTitle_->setObjectName(QStringLiteral("chatTitle"));
    QFont titleFont = chatTitle_->font(); titleFont.setPointSize(20); titleFont.setWeight(QFont::DemiBold); chatTitle_->setFont(titleFont);
    titleLayout->addWidget(chatTitle_, 1);
    modelDescription_ = new QLabel(titleBar);
    modelDescription_->setObjectName(QStringLiteral("modelDescription"));
    modelDescription_->setWordWrap(true);
    titleLayout->addWidget(modelDescription_);
    mainLayout->addWidget(titleBar);

    auto *top = new QHBoxLayout;
    top->setSpacing(8);
    modelSelector_ = new QComboBox(main);
    modelSelector_->setObjectName(QStringLiteral("modelSelector"));
    modelSelector_->setMinimumWidth(260);
    modelSelector_->setPlaceholderText(QStringLiteral("Scegli un modello"));
    modelSelector_->setAccessibleName(QStringLiteral("Modello attivo"));
    modelSelector_->setEnabled(false);
    catalogButton_ = new QPushButton(QStringLiteral("Modelli"), main);
    catalogButton_->setObjectName(QStringLiteral("catalogButton"));
    catalogButton_->setAccessibleName(QStringLiteral("Modelli"));
    catalogButton_->setToolTip(QStringLiteral("Importa o gestisci il catalogo locale dei modelli"));
    settingsButton_ = new QPushButton(QStringLiteral("Impostazioni"), main);
    settingsButton_->setObjectName(QStringLiteral("settingsButton"));
    settingsButton_->setAccessibleName(QStringLiteral("Impostazioni"));
    top->addWidget(modelSelector_, 1); top->addWidget(catalogButton_); top->addWidget(settingsButton_);
    mainLayout->addLayout(top);
    connect(catalogButton_, &QPushButton::clicked, this, &MainWindow::catalogRequested);
    connect(settingsButton_, &QPushButton::clicked, this, &MainWindow::showSettingsDialog);
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
    emptyState_ = new QLabel(QStringLiteral("Nessun modello installato.\nApri Modelli per configurare un modello locale."), transcriptContent_);
    emptyState_->setObjectName(QStringLiteral("emptyState"));
    emptyState_->setAlignment(Qt::AlignCenter);
    emptyState_->setWordWrap(true);
    emptyState_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    transcriptLayout_->insertWidget(0, emptyState_, 1);
    mainLayout->addWidget(transcriptScroll_, 1);

    auto *composerPanel = new QWidget(main);
    composerPanel->setObjectName(QStringLiteral("composerPanel"));
    auto *composerPanelLayout = new QVBoxLayout(composerPanel);
    composerPanelLayout->setContentsMargins(12, 9, 12, 11);
    composerPanelLayout->setSpacing(7);
    statusLabel_ = new QLabel(QStringLiteral("Pronto"), composerPanel);
    statusLabel_->setObjectName(QStringLiteral("statusLabel"));
    composerPanelLayout->addWidget(statusLabel_);
    auto *composer = new QHBoxLayout;
    composer->setSpacing(10);
    auto *editor = new DraftEditor(composerPanel);
    messageInput_ = editor;
    messageInput_->setObjectName(QStringLiteral("messageInput"));
    messageInput_->setAccessibleName(QStringLiteral("Messaggio"));
    messageInput_->setPlaceholderText(QStringLiteral("Scrivi un messaggio…  (Invio invia, Maiusc+Invio va a capo)"));
    messageInput_->setMinimumHeight(88);
    messageInput_->setMaximumHeight(112);
    editor->send = [this] { submitDraft(); };
    auto *buttons = new QVBoxLayout;
    sendButton_ = new QPushButton(QStringLiteral("Invia"), composerPanel);
    sendButton_->setObjectName(QStringLiteral("sendButton"));
    sendButton_->setAccessibleName(QStringLiteral("Invia messaggio"));
    sendButton_->setEnabled(false);
    sendButton_->setMinimumWidth(100);
    stopButton_ = new QPushButton(QStringLiteral("Ferma"), composerPanel);
    stopButton_->setObjectName(QStringLiteral("stopButton"));
    stopButton_->setAccessibleName(QStringLiteral("Interrompi generazione"));
    stopButton_->setEnabled(false);
    stopButton_->setMinimumWidth(100);
    stopButton_->setVisible(false);
    buttons->addWidget(sendButton_); buttons->addWidget(stopButton_); buttons->addStretch();
    composer->addWidget(messageInput_, 1); composer->addLayout(buttons);
    composerPanelLayout->addLayout(composer);
    mainLayout->addWidget(composerPanel);
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
                (!model.approved ? QStringLiteral("Non approvato") : QStringLiteral("Non disponibile"));
            label += QStringLiteral(" — ") + reason;
        }
        modelSelector_->addItem(label, model.id);
        const int row = modelSelector_->count() - 1;
        modelSelector_->setItemData(row, usable, Qt::UserRole + 1);
        modelSelector_->setItemData(row, model.unavailableReason, Qt::ToolTipRole);
        modelSelector_->setItemData(row, QColor(Theme::palette(themeName_).muted), Qt::ForegroundRole);
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
    modelSelector_->setPlaceholderText(QStringLiteral("Scegli un modello"));
    modelSelector_->blockSignals(false);
    if (!hasAvailable) {
        emptyState_->setText(QStringLiteral("Nessun modello installato.\nApri Modelli per configurare un modello locale."));
    } else {
        emptyState_->setText(QStringLiteral("Scegli un modello per iniziare una chat."));
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
    if (hasConversation_ && !currentId.isEmpty()) {
        for (const auto &conversation : conversations_) {
            if (conversation.id == currentId) {
                chatTitle_->setText(conversation.title.trimmed().isEmpty() ? QStringLiteral("Nuova chat") : conversation.title);
                break;
            }
        }
    }
}

void MainWindow::showConversation(const Conversation &conversation) {
    currentConversation_ = conversation;
    hasConversation_ = !conversation.id.isEmpty();
    if (!hasConversation_) currentConversation_ = Conversation{};
    chatTitle_->setText(hasConversation_ && !conversation.title.trimmed().isEmpty()
        ? conversation.title : QStringLiteral("Nuova chat"));
    if (!hasConversation_) {
        conversationList_->blockSignals(true);
        conversationList_->setCurrentRow(-1);
        conversationList_->blockSignals(false);
    }
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
        if (item->widget() && item->widget() != emptyState_) delete item->widget();
        delete item;
    }
    transcriptLayout_->insertWidget(0, emptyState_, 1);
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
    const bool userMessage = message.role.compare(QStringLiteral("assistant"), Qt::CaseInsensitive) != 0;
    card->setProperty("userMessage", userMessage);
    card->setProperty("assistantMessage", !userMessage);
    card->setMaximumWidth(790);
    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(14, 12, 14, 14);
    layout->setSpacing(7);
    auto *header = new QHBoxLayout;
    auto *role = new QLabel(userMessage ? QStringLiteral("Tu") : QStringLiteral("SiliconLLM"), card);
    role->setObjectName(QStringLiteral("messageRole"));
    QFont f = role->font(); f.setWeight(QFont::DemiBold); role->setFont(f);
    header->addWidget(role); header->addStretch();
    auto *copy = new QPushButton(QStringLiteral("Copia"), card);
    copy->setObjectName(QStringLiteral("copyMessageButton"));
    copy->setMaximumWidth(70);
    connect(copy, &QPushButton::clicked, this, [text = message.content] { QApplication::clipboard()->setText(text); });
    header->addWidget(copy);
    layout->addLayout(header);
    auto *body = makeMarkdownView(message.content, card, themeName_, textSize_);
    body->setMinimumWidth(0);
    if (draft) {
        body->setHtml(QString());
        body->setPlainText(message.content);
        draftBrowser_ = body;
    }
    body->setMinimumHeight(qMin(450, qMax(38, qCeil(body->document()->size().height() + 10))));
    layout->addWidget(body);
    if (!draft && !userMessage && showStatistics_) {
        const QString stats = statisticsText(message.statistics);
        if (!stats.isEmpty()) {
            auto *details = new QLabel(stats, card);
            details->setObjectName(QStringLiteral("messageStatistics"));
            details->setWordWrap(true);
            layout->addWidget(details);
        }
    }
    auto *row = new QWidget(transcriptContent_);
    auto *rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->addStretch(1);
    rowLayout->addWidget(card, 100);
    rowLayout->addStretch(1);
    row->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    transcriptLayout_->insertWidget(transcriptLayout_->count() - 1, row);

    static const QRegularExpression codePattern(QStringLiteral("```[^\\n]*\\n([\\s\\S]*?)```"));
    auto matches = codePattern.globalMatch(message.content);
    int index = 0;
    while (matches.hasNext()) {
        const QString code = matches.next().captured(1);
        auto *copyCode = new QPushButton(QStringLiteral("Copia codice %1").arg(++index), card);
        copyCode->setObjectName(QStringLiteral("copyCodeButton"));
        copyCode->setMaximumWidth(120);
        connect(copyCode, &QPushButton::clicked, this, [code] { QApplication::clipboard()->setText(code); });
        layout->addWidget(copyCode, 0, Qt::AlignRight);
    }
}

void MainWindow::setBusy(bool busy, const QString &status) {
    if (busy && !busy_) firstContentDisplayedForRequest_ = false;
    busy_ = busy;
    statusLabel_->setText(status);
    statusLabel_->setProperty("error", false);
    statusLabel_->style()->unpolish(statusLabel_);
    statusLabel_->style()->polish(statusLabel_);
    updateSendAvailability();
    stopButton_->setEnabled(busy);
    sendButton_->setVisible(!busy);
    stopButton_->setVisible(busy);
    catalogButton_->setEnabled(!busy);
    settingsButton_->setEnabled(!busy);
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
    const bool hasUsableSelectedModel = index >= 0 && modelSelector_->itemData(index, Qt::UserRole + 1).toBool();
    bool canSend = hasUsableSelectedModel;
    if (hasConversation_)
        canSend = canSend && modelSelector_->itemData(index).toString() == currentConversation_.modelId;
    sendButton_->setEnabled(!busy_ && canSend);
    newChatButton_->setEnabled(!busy_ && hasUsableSelectedModel);
}

void MainWindow::setDraftResponse(const QString &cumulativeText) {
    draft_ = cumulativeText;
    if (!draftBrowser_ && !draft_.isEmpty()) renderConversation();
    if (busy_ && !draft_.isEmpty() && !firstContentDisplayedForRequest_) {
        draftTimer_.stop();
        updateDraftView(false);
    } else if (busy_) {
        if (!draftTimer_.isActive()) draftTimer_.start();
    }
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
    statusLabel_->setText(QStringLiteral("Errore: ") + message);
    statusLabel_->setProperty("error", true);
    statusLabel_->style()->unpolish(statusLabel_);
    statusLabel_->style()->polish(statusLabel_);
}

QString MainWindow::theme() const
{
    return themeName_;
}

void MainWindow::setTheme(const QString &name)
{
    const QString normalized = Theme::normalize(name);
    if (themeName_ == normalized) return;
    themeName_ = normalized;
    QSettings settings(QStringLiteral("SiliconLLM"), QStringLiteral("LocalChat"));
    settings.setValue(QStringLiteral("ui/theme"), themeName_);
    settings.sync();
    applyAppearance(true);
    emit appearanceChanged();
}

void MainWindow::applyAppearance(bool rerenderTranscript)
{
    setStyleSheet(Theme::styleSheet(themeName_, textSize_, staticGlass_));
#ifdef Q_OS_WIN
    const BOOL darkTitleBar = themeName_ == QStringLiteral("dark") ? TRUE : FALSE;
    constexpr DWORD useImmersiveDarkMode = 20;
    DwmSetWindowAttribute(reinterpret_cast<HWND>(winId()),
                          static_cast<DWMWINDOWATTRIBUTE>(useImmersiveDarkMode),
                          &darkTitleBar, sizeof(darkTitleBar));
#endif
    QPalette palette = QApplication::palette();
    const ThemePalette colors = Theme::palette(themeName_);
    palette.setColor(QPalette::Window, QColor(colors.window));
    palette.setColor(QPalette::WindowText, QColor(colors.text));
    palette.setColor(QPalette::Base, QColor(colors.surface));
    palette.setColor(QPalette::AlternateBase, QColor(colors.sidebar));
    palette.setColor(QPalette::Text, QColor(colors.text));
    palette.setColor(QPalette::Button, QColor(colors.surface));
    palette.setColor(QPalette::ButtonText, QColor(colors.text));
    palette.setColor(QPalette::Highlight, QColor(colors.selection));
    palette.setColor(QPalette::HighlightedText, QColor(colors.text));
    setPalette(palette);
    for (int i = 0; i < modelSelector_->count(); ++i)
        if (!modelSelector_->itemData(i, Qt::UserRole + 1).toBool())
            modelSelector_->setItemData(i, QColor(colors.muted), Qt::ForegroundRole);
    if (rerenderTranscript) renderConversation();
}

void MainWindow::showSettingsDialog()
{
    if (busy_) return;
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("settingsDialog"));
    dialog.setWindowTitle(QStringLiteral("Impostazioni"));
    dialog.setMinimumWidth(380);
    auto *layout = new QVBoxLayout(&dialog);
    auto *form = new QFormLayout;

    auto *themeChoice = new QComboBox(&dialog);
    themeChoice->setObjectName(QStringLiteral("themeSelector"));
    themeChoice->addItem(QStringLiteral("Chiaro"), QStringLiteral("light"));
    themeChoice->addItem(QStringLiteral("Scuro"), QStringLiteral("dark"));
    themeChoice->setCurrentIndex(themeChoice->findData(themeName_));
    form->addRow(QStringLiteral("Aspetto"), themeChoice);

    auto *sizeChoice = new QComboBox(&dialog);
    sizeChoice->setObjectName(QStringLiteral("textSizeSelector"));
    sizeChoice->addItem(QStringLiteral("Compatto — 11 pt"), 11);
    sizeChoice->addItem(QStringLiteral("Confortevole — 12 pt"), 12);
    sizeChoice->addItem(QStringLiteral("Ampio — 14 pt"), 14);
    sizeChoice->setCurrentIndex(sizeChoice->findData(textSize_));
    form->addRow(QStringLiteral("Testo"), sizeChoice);
    layout->addLayout(form);

    auto *statistics = new QCheckBox(QStringLiteral("Mostra statistiche di generazione"), &dialog);
    statistics->setObjectName(QStringLiteral("statisticsToggle"));
    statistics->setChecked(showStatistics_);
    layout->addWidget(statistics);
    auto *glass = new QCheckBox(QStringLiteral("Bordi chiari sui pannelli"), &dialog);
    glass->setObjectName(QStringLiteral("glassToggle"));
    glass->setChecked(staticGlass_);
    layout->addWidget(glass);
    auto *note = new QLabel(QStringLiteral("Le preferenze vengono salvate su questo computer."), &dialog);
    note->setWordWrap(true);
    note->setMaximumWidth(440);
    layout->addWidget(note);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    buttons->setObjectName(QStringLiteral("settingsButtons"));
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("Applica"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("Annulla"));
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);

    if (dialog.exec() != QDialog::Accepted) return;
    themeName_ = Theme::normalize(themeChoice->currentData().toString());
    textSize_ = sizeChoice->currentData().toInt();
    showStatistics_ = statistics->isChecked();
    staticGlass_ = glass->isChecked();
    QSettings settings(QStringLiteral("SiliconLLM"), QStringLiteral("LocalChat"));
    settings.setValue(QStringLiteral("ui/theme"), themeName_);
    settings.setValue(QStringLiteral("ui/textSize"), textSize_);
    settings.setValue(QStringLiteral("ui/showStatistics"), showStatistics_);
    settings.setValue(QStringLiteral("ui/staticGlass"), staticGlass_);
    settings.sync();
    applyAppearance(true);
    emit appearanceChanged();
}

void MainWindow::showConversationMenu(const QPoint &position)
{
    if (busy_) return;
    auto *item = conversationList_->itemAt(position);
    if (!item) return;
    conversationList_->setCurrentItem(item);
    QMenu menu(this);
    menu.setObjectName(QStringLiteral("conversationContextMenu"));
    QAction *rename = menu.addAction(QStringLiteral("Rinomina…"));
    rename->setObjectName(QStringLiteral("renameConversationAction"));
    QAction *remove = menu.addAction(QStringLiteral("Elimina"));
    remove->setObjectName(QStringLiteral("deleteConversationAction"));
    connect(rename, &QAction::triggered, this, &MainWindow::renameSelectedConversation);
    connect(remove, &QAction::triggered, this, &MainWindow::deleteSelectedConversation);
    menu.exec(conversationList_->viewport()->mapToGlobal(position));
}

void MainWindow::renameSelectedConversation()
{
    auto *item = conversationList_->currentItem();
    if (!item || busy_) return;
    bool accepted = false;
    const QString title = QInputDialog::getText(this, QStringLiteral("Rinomina conversazione"),
        QStringLiteral("Nome della conversazione"), QLineEdit::Normal, item->text(), &accepted).trimmed();
    if (accepted && !title.isEmpty()) emit renameConversationRequested(item->data(Qt::UserRole).toString(), title);
}

void MainWindow::deleteSelectedConversation()
{
    auto *item = conversationList_->currentItem();
    if (item && !busy_) emit deleteConversationRequested(item->data(Qt::UserRole).toString());
}

void MainWindow::updateModelDescription() {
    const QString id = modelSelector_->currentData().toString();
    for (const auto &model : models_) {
        if (model.id != id) continue;
        if (!model.approved || !model.available) {
            modelDescription_->clear();
            modelDescription_->hide();
            return;
        }
        QString text;
        text = model.streaming ? QStringLiteral("Risposta progressiva") : QStringLiteral("Risposta al termine della generazione");
        modelDescription_->setText(text);
        modelDescription_->setToolTip(model.qualification.value(QStringLiteral("reason")).toString());
        modelDescription_->show();
        return;
    }
    modelDescription_->clear();
    modelDescription_->hide();
}

void MainWindow::submitDraft() {
    if (busy_ || !sendButton_->isEnabled()) return;
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
