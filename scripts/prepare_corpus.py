# -*- coding: utf-8 -*-
"""整理各中文语料为按说话人组织的 16kHz 单声道 PCM16 wav + manifest.jsonl。

用法:
    python scripts/prepare_corpus.py --corpus aishell
    python scripts/prepare_corpus.py --corpus stcmds
    python scripts/prepare_corpus.py --corpus primewords
    python scripts/prepare_corpus.py --corpus aidatatang
    python scripts/prepare_corpus.py --corpus all        # 处理全部已解压语料

目录结构假设 (data/extracted/<corpus>/ 下):
  aishell:     data_aishell/wav/{train,dev,test}/S0002/*.wav   (16kHz)
  stcmds:      优先从 data/raw/ms_stcmds/data/*.parquet 解码 (ModelScope 镜像),
               文件名 20170001P00338A0110.wav 中 Pxxxxx 即说话人;
               否则回退 data/extracted/stcmds/ST-CMDS-20170001_1-OS/<speaker_dir>/*.wav
  primewords:  primewords_md_2018_set1/audio_files/<a>/<b>/<uuid>.wav (16kHz)
               speaker 取 set1_transcript.json 中的 user_id
  aidatatang:  aidatatang_200zh/corpus/{train,dev,test}/G0001/*.wav (16kHz)

输出:
  data/speakers/<corpus>/<speaker_id>/<utt_id>.wav
  data/manifest.jsonl  (每行 {"corpus","speaker","path","duration_s"})
过滤 <0.5s 的过短句。已 16k/单声道/PCM16 的文件直接拷贝, 否则重采样转换。
"""
import argparse
import json
import re
import shutil
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "data" / "extracted"
SPEAKERS_OUT = ROOT / "data" / "speakers"
MANIFEST = ROOT / "data" / "manifest.jsonl"
MIN_DUR = 0.5
TARGET_SR = 16000


def spk_aishell(rel: Path) -> str:
    # .../wav/train/S0002/xxx.wav
    for part in rel.parts:
        if re.fullmatch(r"S\d{3,5}", part):
            return part
    return rel.parent.name


def spk_stcmds(rel: Path) -> str:
    # ST-CMDS-20170001_1-OS/<speaker_dir>/xxx.wav — 第一层目录即说话人
    return rel.parts[1] if len(rel.parts) > 2 else rel.parent.name


_PRIMEVORDS_MAP = None


def spk_primewords(rel: Path) -> str:
    # audio_files/0/0a/<uuid>.wav — speaker 来自 set1_transcript.json 的 user_id
    global _PRIMEVORDS_MAP
    if _PRIMEVORDS_MAP is None:
        _PRIMEVORDS_MAP = {}
        for js in (EXTRACTED / "primewords").rglob("set1_transcript.json"):
            with open(js, encoding="utf-8") as f:
                for item in json.load(f):
                    if item.get("file") and item.get("user_id"):
                        _PRIMEVORDS_MAP[Path(item["file"]).stem] = "U" + str(item["user_id"])
    return _PRIMEVORDS_MAP.get(rel.stem, rel.stem[:4])


def spk_aidatatang(rel: Path) -> str:
    for part in rel.parts:
        if re.fullmatch(r"G\d{3,6}", part):
            return part
    return rel.parent.name


CORPORA = {
    "aishell": spk_aishell,
    "stcmds": spk_stcmds,
    "primewords": spk_primewords,
    "aidatatang": spk_aidatatang,
}


def wav_info(path: Path):
    """返回 (sr, channels, frames, subtype_is_pcm16)。失败返回 None。"""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnchannels(), w.getnframes(), w.getsampwidth() == 2
    except wave.Error:
        pass
    # 非标准头 (如带 LIST chunk 的 wav wave 模块也能读, 这里兜底用 soundfile)
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.samplerate, info.channels, info.frames, "PCM" in info.subtype
    except Exception:
        return None


def convert_to_16k(src: Path, dst: Path):
    """用 soundfile 读 + numpy 线性插值重采样到 16k 单声道 PCM16。"""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data[:, 0]
    if sr != TARGET_SR:
        n_out = int(round(len(data) * TARGET_SR / sr))
        if n_out < 1:
            return 0.0
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        data = np.interp(x_new, x_old, data).astype("float32")
    sf.write(str(dst), data, TARGET_SR, subtype="PCM_16")
    return len(data) / TARGET_SR


def process_one(args):
    corpus, src_str, rel_str, out_str = args
    src, rel, out = Path(src_str), Path(rel_str), Path(out_str)
    info = wav_info(src)
    if info is None:
        return None
    sr, ch, frames, pcm16 = info
    if sr <= 0:
        return None
    dur = frames / sr
    if dur < MIN_DUR:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    if sr == TARGET_SR and ch == 1 and pcm16:
        try:
            shutil.copyfile(src, out)
        except OSError:
            return None
    else:
        try:
            dur = convert_to_16k(src, out)
        except Exception:
            return None
        if dur < MIN_DUR:
            out.unlink(missing_ok=True)
            return None
    return {
        "corpus": corpus,
        "speaker": spk_of(corpus, rel),
        "path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "duration_s": round(dur, 3),
    }


def spk_of(corpus: str, rel: Path) -> str:
    return CORPORA[corpus](rel)


def find_wavs(corpus: str):
    base = EXTRACTED / corpus
    if not base.is_dir():
        return None, []
    wavs = sorted(base.rglob("*.wav"))
    return base, wavs


def run_stcmds_parquet():
    """从 ModelScope 镜像的 parquet 分片解码 ST-CMDS。
    文件名 20170001P00338A0110.wav -> 说话人 P00338。"""
    import io
    pq_dir = ROOT / "data" / "raw" / "ms_stcmds" / "data"
    shards = sorted(pq_dir.glob("*.parquet"))
    if not shards:
        return None
    import pyarrow.parquet as pq
    print(f"[stcmds] 从 {len(shards)} 个 parquet 分片解码")
    records = []
    spk_re = re.compile(r"^20170001(P\d{5})")
    n_skip = 0
    for si, shard in enumerate(shards):
        t = pq.read_table(str(shard), columns=["audio"])
        for row in t.to_pylist():
            a = row["audio"]
            name = Path(a["path"]).stem if a.get("path") else None
            m = spk_re.match(name or "")
            if not m:
                n_skip += 1
                continue
            spk = m.group(1)
            data = a["bytes"]
            try:
                with wave.open(io.BytesIO(data), "rb") as w:
                    sr, ch, frames, sw = (w.getframerate(), w.getnchannels(),
                                          w.getnframes(), w.getsampwidth())
            except wave.Error:
                n_skip += 1
                continue
            dur = frames / sr if sr else 0
            if dur < MIN_DUR or sr != TARGET_SR or ch != 1 or sw != 2:
                n_skip += 1
                continue
            out = SPEAKERS_OUT / "stcmds" / spk / (name + ".wav")
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as f:
                    f.write(data)
            records.append({
                "corpus": "stcmds",
                "speaker": spk,
                "path": str(out.relative_to(ROOT)).replace("\\", "/"),
                "duration_s": round(dur, 3),
            })
        print(f"[stcmds] 分片 {si + 1}/{len(shards)} 累计 {len(records)} 条")
    if n_skip:
        print(f"[stcmds] 跳过 {n_skip} 条 (过短/格式不符/文件名无说话人)")
    return records


def run_corpus(corpus: str, workers: int = 8):
    if corpus == "stcmds":
        recs = run_stcmds_parquet()
        if recs is not None:
            return recs
        print("[stcmds] 无 parquet, 回退解压目录")
    base, wavs = find_wavs(corpus)
    if base is None:
        print(f"[{corpus}] data/extracted/{corpus} 不存在, 跳过")
        return []
    print(f"[{corpus}] 发现 {len(wavs)} 个 wav")
    tasks = []
    for w in wavs:
        rel = w.relative_to(base)
        spk = spk_of(corpus, rel)
        out = SPEAKERS_OUT / corpus / spk / (w.stem + ".wav")
        if out.exists():
            continue
        tasks.append((corpus, str(w), str(rel), str(out)))
    print(f"[{corpus}] 待处理 {len(tasks)} 个 (跳过已存在 {len(wavs) - len(tasks)} 个)")
    records = []
    # 已存在的也要进 manifest: 重新扫描输出目录
    done = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(process_one, tasks, chunksize=64):
                done += 1
                if rec:
                    records.append(rec)
                if done % 5000 == 0:
                    print(f"[{corpus}] 进度 {done}/{len(tasks)}")
    # 合并已存在文件的记录
    out_dir = SPEAKERS_OUT / corpus
    have = {(r["speaker"], Path(r["path"]).stem) for r in records}
    if out_dir.is_dir():
        for w in out_dir.rglob("*.wav"):
            key = (w.parent.name, w.stem)
            if key in have:
                continue
            info = wav_info(w)
            if info is None or info[0] != TARGET_SR:
                continue
            dur = info[2] / info[0]
            if dur < MIN_DUR:
                continue
            records.append({
                "corpus": corpus,
                "speaker": w.parent.name,
                "path": str(w.relative_to(ROOT)).replace("\\", "/"),
                "duration_s": round(dur, 3),
            })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA) + ["all"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    all_records = []
    for c in names:
        all_records.extend(run_corpus(c, args.workers))

    # 重写 manifest 中涉及的语料, 保留其它语料的旧记录
    touched = set(names)
    kept = []
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("corpus") not in touched:
                    kept.append(rec)
    kept.extend(all_records)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计
    print("\n===== 统计 (本次处理的语料) =====")
    for c in names:
        recs = [r for r in all_records if r["corpus"] == c]
        spks = {}
        for r in recs:
            spks.setdefault(r["speaker"], []).append(r["duration_s"])
        total_dur = sum(r["duration_s"] for r in recs)
        n_spk = len(spks)
        avg = len(recs) / n_spk if n_spk else 0
        print(f"{c}: 说话人 {n_spk}, 条数 {len(recs)}, "
              f"总时长 {total_dur/3600:.1f}h, 人均 {avg:.0f} 条")
    print(f"manifest: {MANIFEST} (共 {len(kept)} 行)")


if __name__ == "__main__":
    sys.exit(main())
