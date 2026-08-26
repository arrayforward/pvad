// mainwindow.cpp
#include "mainwindow.h"
#include "engine.h"
#include "wizard.h"
#include <QDateTime>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QTimer>
#include <QVBoxLayout>

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
    wizard_btn_ = new QPushButton("开始引导注册", g_enroll);
    enroll_label_ = new QLabel("未注册", g_enroll);
    h1->addWidget(enroll_btn_);
    h1->addWidget(rec_btn_);
    h1->addWidget(clear_btn_);
    h1->addWidget(wizard_btn_);
    h1->addWidget(enroll_label_, 1);
    auto* h1b = new QHBoxLayout();
    rec_state_label_ = new QLabel("", g_enroll);
    auto* hint = new QLabel("建议：每段 3-10 秒，录 3-5 段，变换与麦克风的距离/角度（或用引导注册）", g_enroll);
    hint->setStyleSheet("color: gray;");
    h1b->addWidget(rec_state_label_);
    h1b->addWidget(hint, 1);
    v_enroll->addLayout(h1);
    v_enroll->addLayout(h1b);
    vbox->addWidget(g_enroll);

    // 引导注册面板（默认隐藏）
    wizard_box_ = new QGroupBox("引导注册", this);
    auto* wiz_v = new QVBoxLayout(wizard_box_);
    wiz_step_label_ = new QLabel("第 1/3 段", wizard_box_);
    QFont big = wiz_step_label_->font();
    big.setPointSize(big.pointSize() + 6);
    big.setBold(true);
    wiz_step_label_->setFont(big);
    wiz_hint_label_ = new QLabel(wizard_box_);
    wiz_text_label_ = new QLabel(wizard_box_);
    QFont tfont = wiz_text_label_->font();
    tfont.setPointSize(tfont.pointSize() + 4);
    wiz_text_label_->setFont(tfont);
    wiz_text_label_->setWordWrap(true);
    wiz_text_label_->setStyleSheet("color: #205080;");
    wiz_v->addWidget(wiz_step_label_);
    wiz_v->addWidget(wiz_hint_label_);
    wiz_v->addWidget(wiz_text_label_);
    auto* marks_h = new QHBoxLayout();
    for (int i = 0; i < 3; i++) {
        wiz_marks_[i] = new QLabel(QString("段%1：未录").arg(i + 1), wizard_box_);
        marks_h->addWidget(wiz_marks_[i]);
    }
    marks_h->addStretch(1);
    wiz_v->addLayout(marks_h);
    auto* wiz_btn_h = new QHBoxLayout();
    wiz_rec_btn_ = new QPushButton("开始录音", wizard_box_);
    wiz_sec_label_ = new QLabel("", wizard_box_);
    wiz_cancel_btn_ = new QPushButton("取消向导", wizard_box_);
    wiz_btn_h->addWidget(wiz_rec_btn_);
    wiz_btn_h->addWidget(wiz_sec_label_);
    wiz_btn_h->addStretch(1);
    wiz_btn_h->addWidget(wiz_cancel_btn_);
    wiz_v->addLayout(wiz_btn_h);
    wizard_box_->setVisible(false);
    vbox->addWidget(wizard_box_);

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
    denoise_box_ = new QCheckBox("启用降噪", g_listen);
    wav_pick_btn_ = new QPushButton("选择WAV", g_listen);
    wav_label_ = new QLabel("(未选择)", g_listen);
    listen_btn_ = new QPushButton("开始监听", g_listen);
    stop_btn_ = new QPushButton("停止", g_listen);
    h3->addWidget(mic_radio_);
    h3->addWidget(wav_radio_);
    h3->addWidget(denoise_box_);
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
    connect(wizard_btn_, &QPushButton::clicked, this, [this]() {
        QMetaObject::invokeMethod(engine_, "startWizard", Qt::QueuedConnection);
    });
    connect(wiz_rec_btn_, &QPushButton::clicked, this, [this]() {
        if (recording_)
            QMetaObject::invokeMethod(engine_, "stopRecord", Qt::QueuedConnection);
        else
            QMetaObject::invokeMethod(engine_, "startRecord", Qt::QueuedConnection);
    });
    connect(wiz_cancel_btn_, &QPushButton::clicked, this, [this]() {
        QMetaObject::invokeMethod(engine_, "cancelWizard", Qt::QueuedConnection);
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
    connect(denoise_box_, &QCheckBox::toggled, this, [this](bool on) {
        QMetaObject::invokeMethod(engine_, "setDenoiseEnabled", Qt::QueuedConnection, Q_ARG(bool, on));
    });
    denoise_box_->setChecked(true);  // 降噪默认开（触发 toggled -> engine 创建 Denoise）

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
        if (!on) { rec_state_label_->clear(); rec_sec_ = 0.0; }
        updateButtons();
    });
    connect(engine_, &Engine::recordProgress, this, [this](double sec) {
        rec_sec_ = sec;
        rec_state_label_->setText(QString("录音中 %1s / 15s").arg(sec, 0, 'f', 1));
        if (wizard_mode_ && recording_) {
            wiz_sec_label_->setText(QString("录音中 %1s").arg(sec, 0, 'f', 1));
            // ≥3s 才允许手动停止
            wiz_rec_btn_->setEnabled(sec >= 3.0);
        }
    });
    // ---- 引导注册 ----
    connect(engine_, &Engine::wizardStateChanged, this, [this](bool on) {
        wizard_mode_ = on;
        wizard_box_->setVisible(on);
        if (!on) {
            for (int i = 0; i < 3; i++) {
                wiz_marks_[i]->setText(QString("段%1：未录").arg(i + 1));
                wiz_marks_[i]->setStyleSheet("");
            }
            wiz_sec_label_->clear();
        }
        updateButtons();
    });
    connect(engine_, &Engine::wizardStepChanged, this, &MainWindow::updateWizardStep);
    connect(engine_, &Engine::wizardSegmentAccepted, this, [this](int idx, double sec) {
        if (idx >= 0 && idx < 3) {
            wiz_marks_[idx]->setText(QString("段%1：✓ 已录入（%2s）").arg(idx + 1).arg(sec, 0, 'f', 1));
            wiz_marks_[idx]->setStyleSheet("color: green;");
        }
        wiz_sec_label_->clear();
    });
    connect(engine_, &Engine::wizardSegmentRejected, this, [this](double sec) {
        wiz_sec_label_->setText(QString("太短（%1s），请重录本段").arg(sec, 0, 'f', 1));
    });
    connect(engine_, &Engine::wizardFinished, this, [this](int n) {
        wiz_sec_label_->setText(QString("注册完成（%1 段）✓").arg(n));
    });
}

void MainWindow::updateWizardStep(int step) {
    if (step < 0 || step >= WizardController::kTotal) return;
    wiz_step_label_->setText(QString("第 %1/3 段").arg(step + 1));
    wiz_hint_label_->setText(QString::fromUtf8(WizardController::kSteps[step].hint));
    wiz_text_label_->setText(QString::fromUtf8(WizardController::kSteps[step].text));
    wiz_sec_label_->clear();
}

void MainWindow::updateButtons() {
    bool busy = listening_ || recording_ || wizard_mode_;
    listen_btn_->setEnabled(!busy);
    stop_btn_->setEnabled(listening_);
    rec_btn_->setEnabled(!listening_ && !wizard_mode_);
    rec_btn_->setText(recording_ && !wizard_mode_ ? "停止录音" : "录音注册");
    enroll_btn_->setEnabled(!busy);
    clear_btn_->setEnabled(!busy);
    wizard_btn_->setEnabled(!busy);
    wav_pick_btn_->setEnabled(!busy);
    speak_btn_->setEnabled(!wizard_mode_);
    mic_radio_->setEnabled(!busy);
    wav_radio_->setEnabled(!busy);
    // 向导面板内：录音按钮在录音中需 ≥3s 才能点（由 recordProgress 控制），非录音时常开
    if (wizard_mode_) {
        wiz_rec_btn_->setText(recording_ ? "停止录音" : "开始录音");
        wiz_rec_btn_->setEnabled(!recording_ || rec_sec_ >= 3.0);
        wiz_cancel_btn_->setEnabled(!recording_);
    }
}

MainWindow::~MainWindow() {
    thread_->quit();
    thread_->wait(3000);
}

void MainWindow::log(const QString& s) {
    log_edit_->appendPlainText(QDateTime::currentDateTime().toString("[HH:mm:ss] ") + s);
}
