Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono))
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$outdir = 'D:/ai-app/double-voice/test_audio/cohort'
$zh = '今天我们去超市买了很多东西，晚上一起做饭吃，顺便聊聊最近的工作安排。'
$en = 'The quick brown fox jumps over the lazy dog while the birds are singing in the tall trees outside.'
foreach ($v in @('Microsoft Huihui Desktop','Microsoft Zira Desktop')) {
  if ($v -like '*Huihui*') { $tag='zh'; $txt=$zh; $lang='zh-CN' } else { $tag='en'; $txt=$en; $lang='en-US' }
  $s.SelectVoice($v)
  foreach ($r in @(-8,-4,4,8)) {
    $s.Rate = $r
    $s.SetOutputToWaveFile("$outdir/${tag}_rate${r}.wav", $fmt)
    $s.Speak($txt)
  }
  $s.Rate = 0
  foreach ($p in @('+25%','-25%','+15%','-15%')) {
    $pp = $p.Replace('+','p').Replace('-','m').Replace('%','')
    $s.SetOutputToWaveFile("$outdir/${tag}_pitch${pp}.wav", $fmt)
    $ssml = "<speak version='1.0' xml:lang='$lang'><prosody pitch='$p'>$txt</prosody></speak>"
    $s.SpeakSsml($ssml)
  }
}
$s.SetOutputToNull(); $s.Dispose()
Write-Output 'done'
