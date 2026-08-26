// mainwindow.h
#pragma once
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

    QThread* thread_;
    Engine* engine_;
    QPushButton* enroll_btn_;
    QPushButton* rec_btn_;
    QPushButton* clear_btn_;
    QLabel* rec_state_label_;
    QLabel* enroll_label_;
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
};
