#include "theme.h"

namespace silicon {

QString Theme::normalize(const QString &name)
{
    return name.compare(QStringLiteral("dark"), Qt::CaseInsensitive) == 0
        ? QStringLiteral("dark") : QStringLiteral("light");
}

ThemePalette Theme::palette(const QString &name)
{
    if (normalize(name) == QStringLiteral("dark")) {
        return {
            QStringLiteral("#171918"), QStringLiteral("#1c1f1d"), QStringLiteral("#202321"),
            QStringLiteral("#252a27"), QStringLiteral("#eef0ec"), QStringLiteral("#a4aba5"),
            QStringLiteral("#383e39"), QStringLiteral("#4b554e"), QStringLiteral("#9cc7a8"),
            QStringLiteral("#19201b"), QStringLiteral("#b2d7bd"), QStringLiteral("#2b3930"),
            QStringLiteral("#282d29"), QStringLiteral("#252a26"), QStringLiteral("#151817"),
            QStringLiteral("#f09c8f")
        };
    }
    return {
        QStringLiteral("#f4f4f1"), QStringLiteral("#efefeb"), QStringLiteral("#ffffff"),
        QStringLiteral("#fafaf7"), QStringLiteral("#181916"), QStringLiteral("#666960"),
        QStringLiteral("#d9dbd4"), QStringLiteral("#fbfcf8"), QStringLiteral("#244b3a"),
        QStringLiteral("#ffffff"), QStringLiteral("#2d5b47"), QStringLiteral("#dfe9e1"),
        QStringLiteral("#f1f3ee"), QStringLiteral("#f0f1ed"), QStringLiteral("#f3f3ef"),
        QStringLiteral("#9a382b")
    };
}

QString Theme::styleSheet(const QString &name, int textSize, bool staticGlass)
{
    const ThemePalette c = palette(name);
    const QString edge = staticGlass ? c.edge : c.border;
    return QStringLiteral(
        "QMainWindow, QWidget#windowRoot, QWidget#mainPane { background:%1; }"
        "QWidget { color:%2; font-family:'Segoe UI'; font-size:%3pt; }"
        "QWidget#sidebar { background:%4; border-right:1px solid %5; }"
        "QWidget#mainPane { background:%1; }"
        "QScrollArea, QWidget#chatTranscript { background:%1; border:0; }"
        "QLabel#brandLabel { color:%2; font-size:23pt; font-weight:600; }"
        "QLabel#brandCaption { color:%9; font-size:8pt; }"
        "QLabel#chatTitle { color:%2; font-size:20pt; font-weight:600; }"
        "QWidget#topBar, QWidget#composerPanel { background:%6; border:1px solid %5; border-top:1px solid %7; border-radius:12px; }"
        "QComboBox, QPlainTextEdit { background:%6; color:%2; border:1px solid %5; border-radius:9px; padding:8px 10px; selection-background-color:%8; }"
        "QComboBox QAbstractItemView { background:%6; color:%2; border:1px solid %5; selection-background-color:%8; }"
        "QComboBox:hover, QPlainTextEdit:focus { border-color:%7; }"
        "QComboBox:focus, QPushButton:focus { border:2px solid %10; }"
        "QListWidget { background:transparent; border:0; outline:0; }"
        "QListWidget::item { background:transparent; border-radius:7px; padding:10px 9px; }"
        "QListWidget::item:hover { background:%6; } QListWidget::item:selected { background:%8; color:%2; }"
        "QPushButton { background:%6; color:%2; border:1px solid %5; border-radius:9px; padding:8px 12px; }"
        "QPushButton:hover { background:%8; border-color:%7; } QPushButton:pressed { background:%4; }"
        "QPushButton:disabled { color:%9; background:%4; border-color:%5; }"
        "QPushButton#sendButton { background:%10; color:%11; border-color:%10; font-weight:600; }"
        "QPushButton#sendButton:hover { background:%12; border-color:%12; }"
        "QPushButton#sendButton:disabled { background:%4; color:%9; border-color:%5; }"
        "QPushButton#stopButton { background:%13; color:%14; border-color:%13; }"
        "QPushButton#stopButton:disabled { background:%4; color:%9; border-color:%5; }"
        "QLabel#statusLabel, QLabel#modelDescription, QLabel#messageStatistics { color:%9; }"
        "QLabel#statusLabel[error=\"true\"] { color:%14; }"
        "QLabel#emptyState { color:%9; }"
        "QWidget#messageCard { background:transparent; border:0; border-bottom:1px solid %5; }"
        "QWidget#messageCard[userMessage=\"true\"] { background:%15; border-radius:8px; padding:2px; }"
        "QWidget#messageCard[assistantMessage=\"true\"] { background:transparent; }"
        "QTextBrowser#messageMarkdown { background:transparent; color:%2; border:0; selection-background-color:%8; }"
        "QLabel#messageStatistics { background:%16; border-radius:7px; padding:6px 8px; }"
        "QMenu { background:%6; color:%2; border:1px solid %5; padding:4px; }"
        "QMenu::item { padding:7px 22px; border-radius:5px; } QMenu::item:selected { background:%8; }"
        "QDialog { background:%1; color:%2; } QCheckBox { spacing:8px; }"
        "QScrollBar:vertical { background:transparent; width:10px; margin:2px; }"
        "QScrollBar::handle:vertical { background:%5; border-radius:4px; min-height:28px; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }"
    ).arg(c.window).arg(c.text).arg(QString::number(textSize)).arg(c.sidebar)
     .arg(c.border).arg(c.surface).arg(edge).arg(c.selection).arg(c.muted)
     .arg(c.accent).arg(c.accentText).arg(c.accentHover).arg(c.statistics)
     .arg(c.error).arg(c.userMessage).arg(c.statistics);
}

QString Theme::markdownStyle(const QString &name)
{
    const ThemePalette c = palette(name);
    return QStringLiteral(
        "body { color:%1; } a { color:%2; } "
        "pre, code { background:%3; color:%1; border-radius:5px; } "
        "blockquote { color:%4; border-left:3px solid %5; margin-left:0; padding-left:10px; } "
        "hr { color:%5; background:%5; }")
        .arg(c.text, c.accent, c.code, c.muted, c.border);
}

} // namespace silicon
