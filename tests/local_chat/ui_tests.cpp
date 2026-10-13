#include "../../apps/local-chat/src/mainwindow.h"

#include <QComboBox>
#include <QApplication>
#include <QAction>
#include <QCheckBox>
#include <QContextMenuEvent>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFont>
#include <QLabel>
#include <QImage>
#include <QListWidget>
#include <QMenu>
#include <QMap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QTextDocument>
#include <QTextCursor>
#include <QTest>
#include <QSignalSpy>
#include <QSettings>
#include <QSet>
#include <QTimer>
#include <QUrl>

using namespace silicon;

class MainWindowUiTests final : public QObject {
    Q_OBJECT

private slots:
    void noCatalogDisablesSendAndModelSelection()
    {
        MainWindow window;
        auto *send = window.findChild<QPushButton *>(QStringLiteral("sendButton"));
        auto *selector = window.findChild<QComboBox *>(QStringLiteral("modelSelector"));
        QVERIFY(send);
        QVERIFY(selector);
        QVERIFY(!send->isEnabled());
        QVERIFY(!selector->isEnabled());

        ModelSpec unavailable;
        unavailable.id = QStringLiteral("blocked");
        unavailable.name = QStringLiteral("Blocked model");
        unavailable.unavailableReason = QStringLiteral("Runtime missing");
        window.setModels({unavailable});
        QVERIFY(!send->isEnabled());
        QVERIFY(!selector->isEnabled());
        QVERIFY(!selector->model()->index(0, 0).flags().testFlag(Qt::ItemIsEnabled));
        auto *newChat = window.findChild<QPushButton *>(QStringLiteral("newChatButton"));
        QVERIFY(newChat);
        QVERIFY(!newChat->isEnabled());
    }

    void busyStateLocksNavigationAndLeavesStopAvailable()
    {
        MainWindow window;
        ModelSpec model;
        model.id = QStringLiteral("approved");
        model.name = QStringLiteral("Approved local model");
        model.approved = true;
        model.available = true;
        window.setModels({model});
        window.setConversations({Conversation{QStringLiteral("chat-1"), QStringLiteral("Chat"), model.id, {}, {}}});

        auto *send = window.findChild<QPushButton *>(QStringLiteral("sendButton"));
        auto *stop = window.findChild<QPushButton *>(QStringLiteral("stopButton"));
        auto *selector = window.findChild<QComboBox *>(QStringLiteral("modelSelector"));
        auto *conversations = window.findChild<QListWidget *>(QStringLiteral("conversationList"));
        auto *input = window.findChild<QPlainTextEdit *>(QStringLiteral("messageInput"));
        auto *catalog = window.findChild<QPushButton *>(QStringLiteral("catalogButton"));
        QVERIFY(send && stop && selector && conversations && input && catalog);
        window.setBusy(true, QStringLiteral("Generating"));
        QVERIFY(!send->isEnabled());
        QVERIFY(stop->isEnabled());
        QVERIFY(!selector->isEnabled());
        QVERIFY(!conversations->isEnabled());
        QVERIFY(!input->isEnabled());
        QVERIFY(!catalog->isEnabled());

        window.setBusy(false, QStringLiteral("Ready"));
        QVERIFY(send->isEnabled());
        QVERIFY(!stop->isEnabled());
        QVERIFY(selector->isEnabled());
        QVERIFY(conversations->isEnabled());
        QVERIFY(input->isEnabled());
        QVERIFY(catalog->isEnabled());
    }

    void openingConversationSyncsModelWithoutRequestingChange()
    {
        MainWindow window;
        ModelSpec first;
        first.id = QStringLiteral("model-a");
        first.name = QStringLiteral("Model A");
        first.approved = true;
        first.available = true;
        ModelSpec second = first;
        second.id = QStringLiteral("model-b");
        second.name = QStringLiteral("Model B");
        window.setModels({first, second});
        auto *selector = window.findChild<QComboBox *>(QStringLiteral("modelSelector"));
        auto *send = window.findChild<QPushButton *>(QStringLiteral("sendButton"));
        QVERIFY(selector && send);
        QSignalSpy changed(&window, &MainWindow::modelChangeRequested);

        selector->setCurrentIndex(selector->findData(second.id));
        changed.clear();
        Conversation chat;
        chat.id = QStringLiteral("chat-a");
        chat.modelId = first.id;
        chat.title = QStringLiteral("A conversation");
        window.showConversation(chat);

        QCOMPARE(selector->currentData().toString(), first.id);
        QCOMPARE(changed.count(), 0);
        QVERIFY(send->isEnabled());
    }

    void unavailableConversationRemainsReadableButCannotSend()
    {
        MainWindow window;
        ModelSpec model;
        model.id = QStringLiteral("available-model");
        model.approved = true;
        model.available = true;
        window.setModels({model});

        Conversation chat;
        chat.id = QStringLiteral("old-chat");
        chat.modelId = QStringLiteral("missing-from-catalog");
        Message message;
        message.role = QStringLiteral("assistant");
        message.content = QStringLiteral("Saved response");
        chat.messages.append(message);
        window.showConversation(chat);

        auto *selector = window.findChild<QComboBox *>(QStringLiteral("modelSelector"));
        auto *send = window.findChild<QPushButton *>(QStringLiteral("sendButton"));
        auto *transcript = window.findChild<QWidget *>(QStringLiteral("chatTranscript"));
        QVERIFY(selector && send && transcript);
        QCOMPARE(selector->currentIndex(), -1);
        QVERIFY(selector->isEnabled());
        QVERIFY(!send->isEnabled());
        auto *view = transcript->findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);
        QVERIFY(view->toPlainText().contains(QStringLiteral("Saved response")));
    }

    void emptyConversationClearsActiveChatState()
    {
        MainWindow window;
        ModelSpec model;
        model.id = QStringLiteral("model");
        model.approved = true;
        model.available = true;
        window.setModels({model});
        Conversation chat;
        chat.id = QStringLiteral("chat");
        chat.modelId = model.id;
        window.showConversation(chat);
        window.showConversation(Conversation{});

        auto *send = window.findChild<QPushButton *>(QStringLiteral("sendButton"));
        auto *empty = window.findChild<QLabel *>(QStringLiteral("emptyState"));
        QVERIFY(send && empty);
        QVERIFY(send->isEnabled());
        QVERIFY(!empty->isHidden());
    }

    void enterSendsAndShiftEnterInsertsNewline()
    {
        MainWindow window;
        ModelSpec model;
        model.id = QStringLiteral("approved");
        model.approved = true;
        model.available = true;
        window.setModels({model});
        auto *input = window.findChild<QPlainTextEdit *>(QStringLiteral("messageInput"));
        QVERIFY(input);
        QSignalSpy sent(&window, &MainWindow::sendRequested);

        input->setFocus();
        input->setPlainText(QStringLiteral("hello"));
        QTest::keyClick(input, Qt::Key_Return);
        QCOMPARE(sent.count(), 1);
        QCOMPARE(sent.takeFirst().at(0).toString(), QStringLiteral("hello"));
        QVERIFY(input->toPlainText().isEmpty());

        input->insertPlainText(QStringLiteral("line one"));
        QTest::keyClick(input, Qt::Key_Return, Qt::ShiftModifier);
        input->insertPlainText(QStringLiteral("line two"));
        QCOMPARE(sent.count(), 0);
        QCOMPARE(input->toPlainText(), QStringLiteral("line one\nline two"));
    }

    void streamingKeepsUnicodeAndFormatsWhenFinished()
    {
        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("chat-1");
        window.showConversation(chat);
        window.setBusy(true, QStringLiteral("Generating"));
        const QString partial = QString::fromUtf8("Hello 🌍 — **bold**");
        window.setDraftResponse(partial);
        auto *transcript = window.findChild<QWidget *>(QStringLiteral("chatTranscript"));
        QVERIFY(transcript);
        auto *view = transcript->findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);
        QCOMPARE(view->toPlainText(), partial);

        window.setBusy(false, QStringLiteral("Ready"));
        QTextCursor boldCursor = view->document()->find(QStringLiteral("bold"));
        QVERIFY(!boldCursor.isNull());
        QVERIFY(boldCursor.charFormat().fontWeight() > QFont::Normal);
        QVERIFY(view->toPlainText().contains(QString::fromUtf8("🌍")));
    }

    void firstProgressiveDeltaDisplaysImmediatelyAndSignalsOnlyOnce()
    {
        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("chat-1");
        window.showConversation(chat);
        QSignalSpy displayed(&window, &MainWindow::firstContentDisplayed);
        window.setBusy(true, QStringLiteral("Generating"));

        window.setDraftResponse(QStringLiteral("first"));
        QCOMPARE(displayed.count(), 1);
        auto *transcript = window.findChild<QWidget *>(QStringLiteral("chatTranscript"));
        QVERIFY(transcript);
        auto *view = transcript->findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);
        QCOMPARE(view->toPlainText(), QStringLiteral("first"));

        window.setDraftResponse(QStringLiteral("first second"));
        window.setDraftResponse(QStringLiteral("first second third"));
        QTest::qWait(15);
        QCOMPARE(view->toPlainText(), QStringLiteral("first"));
        QTRY_COMPARE_WITH_TIMEOUT(view->toPlainText(), QStringLiteral("first second third"), 500);
        QCOMPARE(displayed.count(), 1);
    }

    void transcriptUsesReadingWidthAndFitsShortMessages()
    {
        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("layout");
        Message message;
        message.role = QStringLiteral("assistant");
        message.content = QStringLiteral("A short readable reply.");
        chat.messages.append(message);
        window.showConversation(chat);
        window.show();
        auto *view = window.findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);
        QTRY_VERIFY(view->width() > 500);
        QTRY_VERIFY(view->height() < 140);
    }

    void sustainedFastStreamDisplaysBeforeCompletion()
    {
        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("continuous-stream");
        window.showConversation(chat);
        window.setBusy(true, QStringLiteral("Generating"));
        window.setDraftResponse(QStringLiteral("first"));
        auto *view = window.findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);
        for (int index = 0; index < 20; ++index) {
            window.setDraftResponse(QStringLiteral("first %1").arg(index));
            QTest::qWait(15);
        }
        QVERIFY(view->toPlainText() != QStringLiteral("first"));
        QTRY_COMPARE_WITH_TIMEOUT(view->toPlainText(), QStringLiteral("first 19"), 500);
    }

    void markdownCannotLoadLocalImageResources()
    {
        QTemporaryDir directory;
        QVERIFY(directory.isValid());
        const QString imagePath = directory.filePath(QStringLiteral("private.png"));
        QImage image(2, 2, QImage::Format_ARGB32);
        image.fill(Qt::red);
        QVERIFY(image.save(imagePath));

        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("chat-1");
        Message message;
        message.role = QStringLiteral("assistant");
        message.content = QStringLiteral("![private](%1)").arg(QUrl::fromLocalFile(imagePath).toString());
        chat.messages.append(message);
        window.showConversation(chat);
        auto *transcript = window.findChild<QWidget *>(QStringLiteral("chatTranscript"));
        QVERIFY(transcript);
        auto *view = transcript->findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(view);

        const QList<QUrl> resources{
            QUrl::fromLocalFile(imagePath),
            QUrl(QStringLiteral("https://example.invalid/model-image.png")),
            QUrl(QStringLiteral("data:image/png;base64,iVBORw0KGgo=")),
        };
        for (const QUrl &resource : resources) {
            const QVariant loaded = view->document()->resource(QTextDocument::ImageResource, resource);
            QVERIFY2(!loaded.isValid() || loaded.isNull(), qPrintable(resource.toString()));
        }
    }

    void themeAndSettingsPersistAndApply()
    {
        QSettings settings(QStringLiteral("SiliconLLM"), QStringLiteral("LocalChat"));
        const QStringList keys{QStringLiteral("ui/theme"), QStringLiteral("ui/textSize"),
                               QStringLiteral("ui/showStatistics"), QStringLiteral("ui/staticGlass")};
        QMap<QString, QVariant> saved;
        QSet<QString> existing;
        for (const QString &key : keys) {
            if (settings.contains(key)) existing.insert(key);
            saved.insert(key, settings.value(key));
        }
        settings.setValue(QStringLiteral("ui/theme"), QStringLiteral("light"));
        settings.setValue(QStringLiteral("ui/textSize"), 11);
        settings.setValue(QStringLiteral("ui/showStatistics"), true);
        settings.setValue(QStringLiteral("ui/staticGlass"), true);
        settings.sync();

        MainWindow window;
        Conversation chat;
        chat.id = QStringLiteral("settings-chat");
        Message reply;
        reply.role = QStringLiteral("assistant");
        reply.content = QStringLiteral("Readable reply");
        reply.statistics = QJsonObject{{QStringLiteral("generated_tokens"), 4},
                                       {QStringLiteral("decode_seconds"), 1.0}};
        chat.messages.append(reply);
        window.showConversation(chat);
        QVERIFY(window.findChild<QLabel *>(QStringLiteral("messageStatistics")));

        window.setTheme(QStringLiteral("dark"));
        QCOMPARE(window.theme(), QStringLiteral("dark"));
        QVERIFY(window.styleSheet().contains(QStringLiteral("#171918")));
        QCOMPARE(settings.value(QStringLiteral("ui/theme")).toString(), QStringLiteral("dark"));

        QTimer::singleShot(0, [] {
            auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget());
            if (!dialog) return;
            auto *theme = dialog->findChild<QComboBox *>(QStringLiteral("themeSelector"));
            auto *size = dialog->findChild<QComboBox *>(QStringLiteral("textSizeSelector"));
            auto *statistics = dialog->findChild<QCheckBox *>(QStringLiteral("statisticsToggle"));
            auto *glass = dialog->findChild<QCheckBox *>(QStringLiteral("glassToggle"));
            auto *buttons = dialog->findChild<QDialogButtonBox *>(QStringLiteral("settingsButtons"));
            if (!theme || !size || !statistics || !glass || !buttons) return;
            theme->setCurrentIndex(theme->findData(QStringLiteral("light")));
            size->setCurrentIndex(size->findData(12));
            statistics->setChecked(false);
            glass->setChecked(false);
            buttons->button(QDialogButtonBox::Ok)->click();
        });
        QTimer::singleShot(1000, [] {
            if (auto *dialog = qobject_cast<QDialog *>(QApplication::activeModalWidget()))
                dialog->reject();
        });
        window.showSettingsDialog();

        QCOMPARE(window.theme(), QStringLiteral("light"));
        QCOMPARE(settings.value(QStringLiteral("ui/theme")).toString(), QStringLiteral("light"));
        QCOMPARE(settings.value(QStringLiteral("ui/textSize")).toInt(), 12);
        QVERIFY(!settings.value(QStringLiteral("ui/showStatistics")).toBool());
        QVERIFY(!settings.value(QStringLiteral("ui/staticGlass")).toBool());
        QVERIFY(!window.findChild<QLabel *>(QStringLiteral("messageStatistics")));
        auto *markdown = window.findChild<QTextBrowser *>(QStringLiteral("messageMarkdown"));
        QVERIFY(markdown);
        QCOMPARE(markdown->font().pointSize(), 12);

        for (const QString &key : keys) {
            if (existing.contains(key)) settings.setValue(key, saved.value(key));
            else settings.remove(key);
        }
        settings.sync();
    }

    void conversationActionsLiveInContextMenu()
    {
        MainWindow window;
        ModelSpec model;
        model.id = QStringLiteral("approved");
        model.approved = true;
        model.available = true;
        window.setModels({model});
        Conversation chat;
        chat.id = QStringLiteral("chat-1");
        chat.title = QStringLiteral("Prima chat");
        chat.modelId = model.id;
        window.setConversations({chat});
        auto *list = window.findChild<QListWidget *>(QStringLiteral("conversationList"));
        QVERIFY(list);
        for (auto *button : window.findChildren<QPushButton *>()) {
            QVERIFY(button->text() != QStringLiteral("Rinomina…"));
            QVERIFY(button->text() != QStringLiteral("Elimina"));
        }

        QSignalSpy deleted(&window, &MainWindow::deleteConversationRequested);
        window.resize(1100, 700);
        window.show();
        QTest::qWait(20);
        const QRect itemRect = list->visualItemRect(list->item(0));
        QVERIFY(itemRect.isValid());
        bool hasRenameAction = false;
        bool hasDeleteAction = false;
        QTimer::singleShot(0, [&] {
            auto *menu = qobject_cast<QMenu *>(QApplication::activePopupWidget());
            if (!menu) return;
            hasRenameAction = menu->findChild<QAction *>(QStringLiteral("renameConversationAction")) != nullptr;
            auto *remove = menu->findChild<QAction *>(QStringLiteral("deleteConversationAction"));
            hasDeleteAction = remove != nullptr;
            if (remove) remove->trigger();
            menu->close();
        });
        QTimer::singleShot(1000, [] {
            if (auto *popup = QApplication::activePopupWidget()) popup->close();
        });
        const QPoint globalPosition = list->viewport()->mapToGlobal(itemRect.center());
        QContextMenuEvent contextEvent(QContextMenuEvent::Mouse, itemRect.center(), globalPosition, Qt::NoModifier);
        QApplication::sendEvent(list->viewport(), &contextEvent);
        QVERIFY(hasRenameAction);
        QVERIFY(hasDeleteAction);
        QCOMPARE(deleted.count(), 1);
        QCOMPARE(deleted.takeFirst().at(0).toString(), chat.id);
    }
};

QTEST_MAIN(MainWindowUiTests)
#include "ui_tests.moc"
