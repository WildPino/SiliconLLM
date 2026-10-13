#pragma once

#include "types.h"
#include "theme.h"

#include <QMainWindow>
#include <QPoint>
#include <QTimer>

class QComboBox;
class QLabel;
class QListWidget;
class QPushButton;
class QPlainTextEdit;
class QScrollArea;
class QVBoxLayout;
class QWidget;
class QEvent;
class QTextBrowser;

namespace silicon {

class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

    void setModels(const QList<ModelSpec> &models);
    void setConversations(const QList<Conversation> &conversations);
    void showConversation(const Conversation &conversation);
    void setBusy(bool busy, const QString &status);
    void setDraftResponse(const QString &cumulativeText);
    void showError(const QString &message);
    void setTheme(const QString &name);
    QString theme() const;
    void showSettingsDialog();

signals:
    void newConversationRequested(const QString &modelId);
    void conversationSelected(const QString &id);
    void renameConversationRequested(const QString &id, const QString &title);
    void deleteConversationRequested(const QString &id);
    void sendRequested(const QString &text);
    void stopRequested();
    void modelChangeRequested(const QString &modelId);
    void catalogRequested();
    void firstContentDisplayed();
    void appearanceChanged();

private:
    bool eventFilter(QObject *watched, QEvent *event) override;
    void buildUi();
    void renderConversation();
    void appendMessage(const Message &message, bool draft = false);
    void clearTranscript();
    void scrollToBottom();
    void updateModelDescription();
    void updateSendAvailability();
    void updateDraftView(bool finalize);
    void submitDraft();
    void applyAppearance(bool rerenderTranscript);
    void showConversationMenu(const QPoint &position);
    void renameSelectedConversation();
    void deleteSelectedConversation();

    QList<ModelSpec> models_;
    QList<Conversation> conversations_;
    Conversation currentConversation_;
    bool hasConversation_ = false;
    bool busy_ = false;
    bool firstContentDisplayedForRequest_ = false;
    bool showStatistics_ = true;
    int textSize_ = 11;
    QString themeName_ = QStringLiteral("light");
    QString draft_;

    QComboBox *modelSelector_ = nullptr;
    QListWidget *conversationList_ = nullptr;
    QPlainTextEdit *messageInput_ = nullptr;
    QPushButton *sendButton_ = nullptr;
    QPushButton *stopButton_ = nullptr;
    QPushButton *newChatButton_ = nullptr;
    QPushButton *catalogButton_ = nullptr;
    QPushButton *settingsButton_ = nullptr;
    QWidget *transcriptContent_ = nullptr;
    QVBoxLayout *transcriptLayout_ = nullptr;
    QScrollArea *transcriptScroll_ = nullptr;
    QLabel *statusLabel_ = nullptr;
    QLabel *emptyState_ = nullptr;
    QLabel *modelDescription_ = nullptr;
    QLabel *chatTitle_ = nullptr;
    QTextBrowser *draftBrowser_ = nullptr;
    QTimer draftTimer_;
};

} // namespace silicon
