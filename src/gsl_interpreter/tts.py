from __future__ import annotations

import asyncio
import ctypes
from dataclasses import dataclass
from hashlib import sha1
import platform
from pathlib import Path
import queue
import threading
import time
from typing import Iterable
import webbrowser

DEFAULT_GEORGIAN_VOICE = "ka-GE-EkaNeural"
MALE_GEORGIAN_VOICE = "ka-GE-GiorgiNeural"
DEFAULT_TTS_MODE = "words"
TTS_MODE_CHOICES = ("off", "saved", "words", "all")


@dataclass(frozen=True)
class TTSQueueResult:
    queued: bool
    message: str


@dataclass(frozen=True)
class TTSRequest:
    texts: tuple[str, ...]
    event: str


class GeorgianTTS:
    def __init__(
        self,
        mode: str = DEFAULT_TTS_MODE,
        voice: str = DEFAULT_GEORGIAN_VOICE,
        rate: str = "+0%",
        volume: str = "+0%",
        cache_dir: str | None = "data/tts",
    ) -> None:
        if mode not in TTS_MODE_CHOICES:
            raise ValueError(f"Unsupported TTS mode: {mode}")
        self.mode = mode
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/tts")
        self._queue: queue.Queue[TTSRequest] = queue.Queue(maxsize=4)
        self._status = "TTS off" if mode == "off" else "TTS ready"
        self._available: bool | None = None
        self._worker: threading.Thread | None = None
        self._prewarm_worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._synthesis_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def status(self) -> str:
        with self._lock:
            return self._status

    def speak_word(self, word: str) -> TTSQueueResult:
        if self.mode not in {"words", "all"}:
            return TTSQueueResult(False, "")
        return self._enqueue(word, "word")

    def speak_sentence(self, sentence: str, event: str) -> TTSQueueResult:
        if self.mode not in {"saved", "all"}:
            return TTSQueueResult(False, "")
        return self._enqueue(sentence, event)

    def speak_sentence_words(self, words: Iterable[str], event: str) -> TTSQueueResult:
        if self.mode not in {"saved", "all"}:
            return TTSQueueResult(False, "")
        texts = tuple(_normalize_text(word) for word in words)
        texts = tuple(text for text in texts if text)
        return self._enqueue_texts(texts, event)

    def prewarm(self, texts: Iterable[str]) -> TTSQueueResult:
        if not self.enabled:
            return TTSQueueResult(False, "")
        normalized = tuple(dict.fromkeys(_normalize_text(text) for text in texts if _normalize_text(text)))
        if not normalized:
            return TTSQueueResult(False, "")
        if not self._ensure_available():
            return TTSQueueResult(False, self.status())
        if self._prewarm_worker and self._prewarm_worker.is_alive():
            return TTSQueueResult(False, "TTS cache already warming")

        self._prewarm_worker = threading.Thread(
            target=self._prewarm_texts,
            args=(normalized,),
            name="gsl-tts-prewarm",
            daemon=True,
        )
        self._prewarm_worker.start()
        self._set_status("TTS warming cache")
        return TTSQueueResult(True, "TTS warming cache")

    def _enqueue(self, text: str, event: str) -> TTSQueueResult:
        text = _normalize_text(text)
        return self._enqueue_texts((text,) if text else (), event)

    def _enqueue_texts(self, texts: tuple[str, ...], event: str) -> TTSQueueResult:
        if not texts:
            return TTSQueueResult(False, "")
        if not self.enabled:
            return TTSQueueResult(False, "")
        if not self._ensure_available():
            return TTSQueueResult(False, self.status())

        self._ensure_worker()
        request = TTSRequest(texts=texts, event=event)
        while True:
            try:
                self._queue.put_nowait(request)
                self._set_status("TTS queued")
                return TTSQueueResult(True, "TTS queued")
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    return TTSQueueResult(False, "TTS queue busy")

    def _ensure_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            self._available = False
            self._set_status("Install edge-tts for Georgian speech")
        else:
            self._available = True
        return self._available

    def _ensure_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run_worker, name="gsl-tts", daemon=True)
        self._worker.start()

    def _run_worker(self) -> None:
        while True:
            request = self._queue.get()
            try:
                self._set_status(f"Speaking: {request.event}")
                for text in request.texts:
                    audio_path = self._synthesize(text)
                    self._play(audio_path)
                self._set_status("TTS ready")
            except Exception as exc:  # noqa: BLE001 - keep inference alive if speech fails.
                self._set_status(f"TTS failed: {exc}")
            finally:
                self._queue.task_done()

    def _prewarm_texts(self, texts: tuple[str, ...]) -> None:
        warmed = 0
        try:
            for text in texts:
                while not self._queue.empty():
                    time.sleep(0.05)
                audio_path = self._audio_path(text)
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    continue
                self._set_status(f"Caching TTS {warmed + 1}/{len(texts)}")
                self._synthesize(text)
                warmed += 1
            if self._queue.empty():
                self._set_status("TTS ready")
        except Exception as exc:  # noqa: BLE001 - prewarm must never break inference.
            self._set_status(f"TTS cache failed: {exc}")

    def _audio_path(self, text: str) -> Path:
        digest = sha1(f"{self.voice}|{self.rate}|{self.volume}|{text}".encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{digest}.mp3"

    def _synthesize(self, text: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        audio_path = self._audio_path(text)
        if audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path
        with self._synthesis_lock:
            if audio_path.exists() and audio_path.stat().st_size > 0:
                return audio_path
            asyncio.run(self._save_edge_audio(text, audio_path))
        return audio_path

    async def _save_edge_audio(self, text: str, audio_path: Path) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
        )
        await communicate.save(str(audio_path))

    def _play(self, audio_path: Path) -> None:
        if platform.system() == "Windows":
            _play_with_mci(audio_path)
            return
        webbrowser.open(audio_path.resolve().as_uri())

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _play_with_mci(audio_path: Path) -> None:
    path = str(audio_path.resolve())
    alias = f"gsltts{threading.get_ident()}{time.perf_counter_ns()}"
    _mci(f'open "{path}" type mpegvideo alias {alias}')
    try:
        _mci(f"play {alias} wait")
    finally:
        _mci(f"close {alias}", fail=False)


def _mci(command: str, fail: bool = True) -> str:
    winmm = ctypes.WinDLL("winmm")
    buffer = ctypes.create_unicode_buffer(256)
    error = winmm.mciSendStringW(command, buffer, len(buffer), None)
    if error and fail:
        error_text = ctypes.create_unicode_buffer(256)
        winmm.mciGetErrorStringW(error, error_text, len(error_text))
        raise RuntimeError(error_text.value or f"MCI error {error}")
    return buffer.value
