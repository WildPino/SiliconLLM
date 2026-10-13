#pragma once

#include <QString>

namespace silicon {

struct ThemePalette {
    QString window;
    QString sidebar;
    QString surface;
    QString raised;
    QString text;
    QString muted;
    QString border;
    QString edge;
    QString accent;
    QString accentText;
    QString accentHover;
    QString selection;
    QString userMessage;
    QString statistics;
    QString code;
    QString error;
};

class Theme final {
public:
    static QString normalize(const QString &name);
    static ThemePalette palette(const QString &name);
    static QString styleSheet(const QString &name, int textSize, bool staticGlass);
    static QString markdownStyle(const QString &name);
};

} // namespace silicon
