#include "../../apps/local-chat/src/engine.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSignalSpy>
#include <QTest>
#include <memory>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace silicon;

#ifndef LOCAL_CHAT_FIXTURE
#error LOCAL_CHAT_FIXTURE must point to fixture_engine.py
#endif
#ifndef LOCAL_CHAT_PYTHON
#error LOCAL_CHAT_PYTHON must point to the Python interpreter
#endif

namespace {
const QString fixturePath = QStringLiteral(LOCAL_CHAT_FIXTURE);
const QString pythonPath = QStringLiteral(LOCAL_CHAT_PYTHON);
const QString expectedText = QStringLiteral("Ciao 🌱 — café");

ModelSpec model(QString mode = QStringLiteral("normal"), bool streaming = true)
{
    ModelSpec result;
    result.id = QStringLiteral("fixture-%1-%2").arg(mode, streaming ? "stream" : "plain");
    result.name = QStringLiteral("Deterministic fixture");
    result.protocol = QStringLiteral("silicon-chat-v1");
    result.executable = pythonPath;
    result.arguments = {QStringLiteral("-u"), fixturePath,
                        QStringLiteral("--mode=%1").arg(mode),
                        QStringLiteral("--max-context=4096")};
    result.workingDirectory = QFileInfo(fixturePath).absolutePath();
    result.approved = true;
    result.available = true;
    result.streaming = streaming;
    result.maxContext = 4096;
    result.maxNewTokens = 32;
    result.timeoutSeconds = 4;
    return result;
}

Conversation conversation()
{
    Conversation c;
    c.id = QStringLiteral("fixture-conversation");
    c.messages = {Message{QStringLiteral("user"), QStringLiteral("Saluta in italiano 🌱"), false, {}}};
    return c;
}

bool waitForSignal(QSignalSpy& spy, int timeout = 3000)
{
    return spy.count() > 0 || spy.wait(timeout);
}
}

class EngineTests final : public QObject {
    Q_OBJECT
private slots:
    void streamingCompletesWithUnicodeAndMetrics()
    {
        EngineSession session;
        QSignalSpy deltas(&session, &EngineSession::responseChanged);
        QSignalSpy completed(&session, &EngineSession::completed);
        QSignalSpy failed(&session, &EngineSession::failed);
        QVERIFY(session.generate(model(), conversation(), 16));
        QTRY_VERIFY_WITH_TIMEOUT(deltas.count() > 0, 3000);
        QCOMPARE(completed.count(), 0);
        QVERIFY(waitForSignal(completed));
        QCOMPARE(failed.count(), 0);
        QVERIFY(deltas.count() >= 1);
        QCOMPARE(deltas.first().at(0).toString(), expectedText);
        QCOMPARE(completed.first().at(0).toString(), expectedText);
        QCOMPARE(completed.first().at(2).toBool(), false);
        const auto stats = completed.first().at(1).toJsonObject();
        QCOMPARE(stats.value("generated_tokens").toInt(), 4);
        QCOMPARE(stats.value("termination").toString(), QStringLiteral("eos"));
        QVERIFY(stats.value("first_token_seconds").toDouble() >= 0.0);
        QVERIFY(stats.value("total_seconds").toDouble() >= 0.0);
        QVERIFY(stats.value("decode_seconds").toDouble() >= 0.0);
        QVERIFY(!session.busy());
    }

    void reusesPersistentProcessAndSendsReset()
    {
        EngineSession session;
        QSignalSpy completed(&session, &EngineSession::completed);
        QSignalSpy failed(&session, &EngineSession::failed);
        const auto spec = model();
        QVERIFY(session.generate(spec, conversation(), 16));
        QVERIFY(waitForSignal(completed));
        const qint64 pid = session.processId();
        QVERIFY(pid > 0);
        QVERIFY(session.generate(spec, conversation(), 16));
        QTRY_COMPARE_WITH_TIMEOUT(completed.count(), 2, 3000);
        QCOMPARE(failed.count(), 0);
        QCOMPARE(session.processId(), pid);
        QCOMPARE(completed.at(1).at(0).toString(), expectedText);
    }

    void cancellationStopsProcessAndNextRequestRestarts()
    {
        EngineSession session;
        QSignalSpy deltas(&session, &EngineSession::responseChanged);
        QSignalSpy completed(&session, &EngineSession::completed);
        QVERIFY(session.generate(model(QStringLiteral("slow")), conversation(), 32));
        QTRY_VERIFY_WITH_TIMEOUT(deltas.count() > 0, 3000);
        const qint64 oldPid = session.processId();
        session.cancel();
        QCOMPARE(completed.count(), 1);
        QVERIFY(completed.first().at(2).toBool());
        QVERIFY(session.processId() == 0);
        QVERIFY(session.generate(model(), conversation(), 16));
        QTRY_COMPARE_WITH_TIMEOUT(completed.count(), 2, 3000);
        QVERIFY(session.processId() > 0);
        QVERIFY(session.processId() != oldPid);
        QCOMPARE(completed.at(1).at(0).toString(), expectedText);
    }

    void cancellationKillsWindowsChildProcess()
    {
#ifdef Q_OS_WIN
        EngineSession session;
        QSignalSpy deltas(&session, &EngineSession::responseChanged);
        QVERIFY(session.generate(model(QStringLiteral("child_process")), conversation(), 32));
        QTRY_VERIFY_WITH_TIMEOUT(deltas.count() > 0, 3000);
        bool parsed = false;
        const qint64 childPid = deltas.first().at(0).toString().toLongLong(&parsed);
        QVERIFY(parsed);
        QVERIFY(childPid > 0);
        HANDLE child = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(childPid));
        QVERIFY2(child != nullptr, "could not open fixture child process");
        const DWORD beforeCancel = WaitForSingleObject(child, 0);
        if (beforeCancel != WAIT_TIMEOUT) {
            CloseHandle(child);
            QVERIFY2(false, "fixture child exited before cancellation");
        }

        QVERIFY(session.processId() > 0);
        session.cancel();
        QCOMPARE(session.processId(), qint64(0));
        const DWORD childWait = WaitForSingleObject(child, 5000);
        if (childWait != WAIT_OBJECT_0 && WaitForSingleObject(child, 0) == WAIT_TIMEOUT) {
            TerminateProcess(child, 1);
            WaitForSingleObject(child, 3000);
        }
        CloseHandle(child);
        QCOMPARE(childWait, static_cast<DWORD>(WAIT_OBJECT_0));
#else
        QSKIP("Windows process job ownership is only available on Windows.");
#endif
    }

    void destructorKillsWindowsChildProcess()
    {
#ifdef Q_OS_WIN
        auto session = std::make_unique<EngineSession>();
        QSignalSpy deltas(session.get(), &EngineSession::responseChanged);
        QVERIFY(session->generate(model(QStringLiteral("child_process")), conversation(), 32));
        QTRY_VERIFY_WITH_TIMEOUT(deltas.count() > 0, 3000);
        bool parsed = false;
        const qint64 childPid = deltas.first().at(0).toString().toLongLong(&parsed);
        QVERIFY(parsed);
        QVERIFY(childPid > 0);
        HANDLE child = OpenProcess(SYNCHRONIZE | PROCESS_TERMINATE, FALSE, static_cast<DWORD>(childPid));
        QVERIFY2(child != nullptr, "could not open fixture child process");
        if (WaitForSingleObject(child, 0) != WAIT_TIMEOUT) {
            CloseHandle(child);
            QVERIFY2(false, "fixture child exited before session destruction");
        }

        session.reset();
        const DWORD childWait = WaitForSingleObject(child, 5000);
        if (childWait != WAIT_OBJECT_0 && WaitForSingleObject(child, 0) == WAIT_TIMEOUT) {
            TerminateProcess(child, 1);
            WaitForSingleObject(child, 3000);
        }
        CloseHandle(child);
        QCOMPARE(childWait, static_cast<DWORD>(WAIT_OBJECT_0));
#else
        QSKIP("Windows process job ownership is only available on Windows.");
#endif
    }

    void timeoutStopsRequest()
    {
        EngineSession session;
        auto spec = model(QStringLiteral("slow_timeout"));
        spec.timeoutSeconds = 1;
        QSignalSpy failed(&session, &EngineSession::failed);
        QVERIFY(session.generate(spec, conversation(), 16));
        QVERIFY(waitForSignal(failed, 3000));
        QVERIFY(failed.first().at(0).toString().contains(QStringLiteral("tempo massimo")));
        QVERIFY(!session.busy());
        QCOMPARE(session.processId(), qint64(0));
    }

    void fixtureContextRejectionFailsAndStopsProcess()
    {
        EngineSession session;
        Conversation oversized = conversation();
        oversized.messages[0].content = QString(4097, QLatin1Char('x'));
        QSignalSpy failed(&session, &EngineSession::failed);
        QSignalSpy completed(&session, &EngineSession::completed);
        QVERIFY(session.generate(model(QStringLiteral("context_rejection")), oversized, 16));
        QVERIFY(waitForSignal(failed));
        QVERIFY(failed.first().at(0).toString().contains(QStringLiteral("context limit exceeded")));
        QCOMPARE(completed.count(), 0);
        QVERIFY(!session.busy());
        QCOMPARE(session.processId(), qint64(0));
    }

    void concurrentRequestIsRejectedWithoutDisturbingActiveRequest()
    {
        EngineSession session;
        QSignalSpy failed(&session, &EngineSession::failed);
        QSignalSpy completed(&session, &EngineSession::completed);
        QVERIFY(session.generate(model(QStringLiteral("slow")), conversation(), 32));
        QTRY_VERIFY_WITH_TIMEOUT(session.processId() > 0, 3000);
        QVERIFY(!session.generate(model(), conversation(), 16));
        QCOMPARE(failed.count(), 1);
        QVERIFY(session.busy());
        session.cancel();
        QCOMPARE(completed.count(), 1);
        QVERIFY(completed.first().at(2).toBool());
    }

    void malformedProtocolEventsFailAndStop_data()
    {
        QTest::addColumn<QString>("mode");
        QTest::addColumn<bool>("streaming");
        QTest::newRow("invalid-json") << QStringLiteral("invalid_line") << true;
        QTest::newRow("request-id") << QStringLiteral("mismatched_request_id") << true;
        QTest::newRow("capabilities") << QStringLiteral("capabilities_mismatch") << true;
        QTest::newRow("response-parity") << QStringLiteral("response_mismatch") << true;
        QTest::newRow("token-limit") << QStringLiteral("over_limit") << true;
        QTest::newRow("bad-metrics") << QStringLiteral("bad_metrics") << true;
        QTest::newRow("oversized-event") << QStringLiteral("oversized_event") << true;
    }

    void malformedProtocolEventsFailAndStop()
    {
        QFETCH(QString, mode);
        QFETCH(bool, streaming);
        EngineSession session;
        QSignalSpy failed(&session, &EngineSession::failed);
        QSignalSpy completed(&session, &EngineSession::completed);
        QVERIFY(session.generate(model(mode, streaming), conversation(), 16));
        QVERIFY(waitForSignal(failed, 4000));
        QVERIFY(!failed.first().at(0).toString().isEmpty());
        QVERIFY(!session.busy());
        QCOMPARE(session.processId(), qint64(0));
        if (mode == QStringLiteral("response_mismatch")) {
            QCOMPARE(completed.count(), 1);
            QVERIFY(completed.first().at(2).toBool());
        } else {
            QCOMPARE(completed.count(), 0);
        }
    }

    void nonStreamingHasNoDeltaOrFirstTokenMetric()
    {
        EngineSession session;
        QSignalSpy deltas(&session, &EngineSession::responseChanged);
        QSignalSpy completed(&session, &EngineSession::completed);
        QVERIFY(session.generate(model(QStringLiteral("nonstream"), false), conversation(), 16));
        QVERIFY(waitForSignal(completed));
        QCOMPARE(deltas.count(), 0);
        QCOMPARE(completed.first().at(0).toString(), expectedText);
        QVERIFY(!completed.first().at(1).toJsonObject().contains("first_token_seconds"));
    }
};

QTEST_GUILESS_MAIN(EngineTests)
#include "engine_tests.moc"
