// mainwindow.cpp
#include "mainwindow.h"
#include "engine.h"
#include <QDateTime>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QTimer>
#include <QVBoxLayout>
#include <QGroupBox>

MainWindow::MainWindow() {
    auto* central = new QWidget(this);
    auto* vbox = new QVBoxLayout(central);

    // 注册区
    auto* g_enroll = new QGroupBox("注册", this);
    auto* v_enroll = new QVBoxLayout(g_enroll);
    auto* h1 = new QHBoxLayout();
    enroll_btn_ = new QPushButton("选择WAV注册A", g_enroll);
    rec_btn_ = new QPushButton("录音注册", g_enroll);
    clear_btn_ = new QPushButton("清空注册", g_enroll);
    enroll_label_ = new QLabel("未注册", g_enroll);
    h1->addWidget(enroll_btn_);
    h1->addWidget(rec_btn_);
    h1->addWidget(clear_btn_);
    h1->addWidget(enroll_label_, 1);
    auto* h1b = new QHBoxLayout();
    rec_state_label_ = new QLabel("", g_enroll);
    auto* hint = new QLabel("建议：每段 3-10 秒，录 3-5 段，变换与麦克风的距离/角度", g_enroll);
    hint->setStyleSheet("color: gray;");
    h1b->addWidget(rec_state_label_);
    h1b->addWidget(hint, 1);
    v_enroll->addLayout(h1);
    v_enroll->addLayout(h1b);
    vbox->addWidget(g_enroll);

    // TTS 区
    auto* g_tts = new QGroupBox("TTS", this);
    auto* h2 = new QHBoxLayout(g_tts);
    text_edit_ = new QLineEdit("你好，现在开始为你播报一段消息，请耐心听完，不要打断我。", g_tts);
    speak_btn_ = new QPushButton("朗读", g_tts);
    tts_label_ = new QLabel("", g_tts);
    h2->addWidget(text_edit_, 1);
    h2->addWidget(speak_btn_);
    h2->addWidget(tts_label_);
    vbox->addWidget(g_tts);

    // 监听区
    auto* g_listen = new QGroupBox("监听", this);
    auto* h3 = new QHBoxLayout(g_listen);
    mic_radio_ = new QRadioButton("麦克风", g_listen);
    wav_radio_ = new QRadioButton("WAV 注入", g_listen);
    mic_radio_->setChecked(true);
    wav_pick_btn_ = new QPushButton("选择WAV", g_listen);
    wav_label_ = new QLabel("(未选择)", g_listen);
    listen_btn_ = new QPushButton("开始监听", g_listen);
    stop_btn_ = new QPushButton("停止", g_listen);
    h3->addWidget(mic_radio_);
    h3->addWidget(wav_radio_);
    h3->addWidget(wav_pick_btn_);
    h3->addWidget(wav_label_, 1);
    h3->addWidget(listen_btn_);
    h3->addWidget(stop_btn_);
    vbox->addWidget(g_listen);

    // 状态区
    auto* g_state = new QGroupBox("门控状态", this);
    auto* h4 = new QHBoxLayout(g_state);
    prob_bar_ = new QProgressBar(g_state);
    prob_bar_->setRange(0, 100);
    prob_label_ = new QLabel("P(target)=-", g_state);
    gate_label_ = new QLabel("consec=-", g_state);
    h4->addWidget(prob_bar_, 1);
    h4->addWidget(prob_label_);
    h4->addWidget(gate_label_);
    vbox->addWidget(g_state);

    // 日志区
    log_edit_ = new QPlainTextEdit(this);
    log_edit_->setReadOnly(true);
    vbox->addWidget(log_edit_, 1);

    setCentralWidget(central);
    resize(760, 560);
    setWindowTitle("PVAD 打断 demo");

    // 引擎线程
    thread_ = new QThread(this);
    engine_ = new Engine();
    engine_->moveToThread(thread_);
    connect(thread_, &QThread::started, engine_, &Engine::init);
    connect(thread_, &QThread::finished, engine_, &QObject::deleteLater);
    thread_->start();

    connect(enroll_btn_, &QPushButton::clicked, this, [this]() {
        auto files = QFileDialog::getOpenFileNames(this, "选择注册 WAV（可多选）", QString(),
                                                   "WAV (*.wav)");
        if (!files.isEmpty()) QMetaObject::invokeMethod(engine_, "enrollFiles",
                                                        Qt::QueuedConnection, Q_ARG(QStringList, files));
    });
    connect(rec_btn_, &QPushButton::clicked, this, [this]() {
        if (recording_)
            QMetaObject::invokeMethod(engine_, "stopRecord", Qt::QueuedConnection);
        else
            QMetaObject::invokeMethod(engine_, "startRecord", Qt::QueuedConnection);
    });
    connect(clear_btn_, &QPushButton::clicked, this, [this]() {
        QMetaObject::invokeMethod(engine_, "clearEnroll", Qt::QueuedConnection);
    });
    connect(speak_btn_, &QPushButton::clicked, this, [this]() {
        QMetaObject::invokeMethod(engine_, "speakText", Qt::QueuedConnection,
                                  Q_ARG(QString, text_edit_->text()));
    });
    connect(wav_pick_btn_, &QPushButton::clicked, this, [this]() {
        auto f = QFileDialog::getOpenFileName(this, "选择注入 WAV", QString(), "WAV (*.wav)");
        if (!f.isEmpty()) { inject_wav_ = f; wav_label_->setText(QFileInfo(f).fileName()); wav_radio_->setChecked(true); }
    });
    connect(listen_btn_, &QPushButton::clicked, this, [this]() {
        if (wav_radio_->isChecked() && !inject_wav_.isEmpty())
            QMetaObject::invokeMethod(engine_, "startListenWav", Qt::QueuedConnection, Q_ARG(QString, inject_wav_));
        else
            QMetaObject::invokeMethod(engine_, "startListenMic", Qt::QueuedConnection);
    });
    connect(stop_btn_, &QPushButton::clicked, this, [this]() {
        QMetaObject::invokeMethod(engine_, "stopListen", Qt::QueuedConnection);
    });

    connect(engine_, &Engine::logLine, this, &MainWindow::log);
    connect(engine_, &Engine::enrollStatus, enroll_label_, &QLabel::setText);
    connect(engine_, &Engine::ttsStatus, tts_label_, &QLabel::setText);
    connect(engine_, &Engine::probUpdate, this, [this](float p, int consec) {
        prob_bar_->setValue(int(p * 100));
        prob_label_->setText(QString("P(target)=%1").arg(p, 0, 'f', 3));
        gate_label_->setText(QString("consec=%1").arg(consec));
    });
    connect(engine_, &Engine::interruptFired, this, [this](double t) {
        gate_label_->setText(QString("INTERRUPT @ %1s").arg(t, 0, 'f', 2));
        gate_label_->setStyleSheet("color: red; font-weight: bold;");
        QTimer::singleShot(2000, this, [this]() { gate_label_->setStyleSheet(""); });
    });
    connect(engine_, &Engine::playbackChanged, this, [this](bool playing) {
        tts_label_->setText(playing ? "播放中" : "空闲");
    });
    connect(engine_, &Engine::listenChanged, this, [this](bool on) {
        listening_ = on;
        updateButtons();
    });
    connect(engine_, &Engine::recordStateChanged, this, [this](bool on) {
        recording_ = on;
        if (!on) rec_state_label_->clear();
        updateButtons();
    });
    connect(engine_, &Engine::recordProgress, this, [this](double sec) {
        rec_state_label_->setText(QString("录音中 %1s / 15s").arg(sec, 0, 'f', 1));
    });
}

void MainWindow::updateButtons() {
    listen_btn_->setEnabled(!listening_ && !recording_);
    stop_btn_->setEnabled(listening_);
    rec_btn_->setEnabled(!listening_);
    rec_btn_->setText(recording_ ? "停止录音" : "录音注册");
    enroll_btn_->setEnabled(!listening_ && !recording_);
    clear_btn_->setEnabled(!listening_ && !recording_);
    wav_pick_btn_->setEnabled(!listening_ && !recording_);
}

MainWindow::~MainWindow() {
    thread_->quit();
    thread_->wait(3000);
}

void MainWindow::log(const QString& s) {
    log_edit_->appendPlainText(QDateTime::currentDateTime().toString("[HH:mm:ss] ") + s);
}
