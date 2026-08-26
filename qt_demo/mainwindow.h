// mainwindow.h
#pragma once
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QThread>

class Engine;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();
    ~MainWindow() override;

private:
    void log(const QString& s);
    void updateButtons();
    void updateWizardStep(int step);

    QThread* thread_;
    Engine* engine_;
    QPushButton* enroll_btn_;
    QPushButton* rec_btn_;
    QPushButton* clear_btn_;
    QPushButton* wizard_btn_;
    QLabel* rec_state_label_;
    QLabel* enroll_label_;
    // 引导注册面板
    QGroupBox* wizard_box_;
    QLabel* wiz_step_label_;
    QLabel* wiz_hint_label_;
    QLabel* wiz_text_label_;
    QLabel* wiz_marks_[3];
    QLabel* wiz_sec_label_;
    QPushButton* wiz_rec_btn_;
    QPushButton* wiz_cancel_btn_;
    QLineEdit* text_edit_;
    QPushButton* speak_btn_;
    QLabel* tts_label_;
    QPushButton* listen_btn_;
    QPushButton* stop_btn_;
    QRadioButton* mic_radio_;
    QRadioButton* wav_radio_;
    QPushButton* wav_pick_btn_;
    QLabel* wav_label_;
    QProgressBar* prob_bar_;
    QLabel* prob_label_;
    QLabel* gate_label_;
    QPlainTextEdit* log_edit_;
    QString inject_wav_;
    bool listening_ = false;
    bool recording_ = false;
    bool wizard_mode_ = false;
    double rec_sec_ = 0.0;
};
