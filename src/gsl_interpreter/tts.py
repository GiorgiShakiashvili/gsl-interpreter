from __future__ import annotations

import asyncio
from dataclasses import dataclass
from hashlib import sha1
import platform
from pathlib import Path
import queue
import subprocess
import threading
import webbrowser

DEFAULT_GEORGIAN_VOICE = "ka-GE-EkaNeural"
MALE_GEORGIAN_VOICE = "ka-GE-GiorgiNeural"
TTS_MODE_CHOICES = ("off", "saved", "words", "all")


@dataclass(frozen=True)
class TTSQueueResult:
    queued: bool
    message: str


@dataclass(frozen=True)
class TTSRequest:
    text: str
    event: str


class GeorgianTTS:
    def __init__(
        self,
        mode: str = "saved",
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
        self._lock = threading.Lock()

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

    def _enqueue(self, text: str, event: str) -> TTSQueueResult:
        text = " ".join(text.split())
        if not text:
            return TTSQueueResult(False, "")
        if not self.enabled:
            return TTSQueueResult(False, "")
        if not self._ensure_available():
            return TTSQueueResult(False, self.status())

        self._ensure_worker()
        request = TTSRequest(text=text, event=event)
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
                audio_path = self._synthesize(request.text)
                self._play(audio_path)
                self._set_status("TTS ready")
            except Exception as exc:  # noqa: BLE001 - keep inference alive if speech fails.
                self._set_status(f"TTS failed: {exc}")
            finally:
                self._queue.task_done()

    def _synthesize(self, text: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        digest = sha1(f"{self.voice}|{self.rate}|{self.volume}|{text}".encode("utf-8")).hexdigest()[:20]
        audio_path = self.cache_dir / f"{digest}.mp3"
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
            _play_with_powershell(audio_path)
            return
        webbrowser.open(audio_path.resolve().as_uri())

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status


def _play_with_powershell(audio_path: Path) -> None:
    uri = audio_path.resolve().as_uri()
    script = f"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationCore
$player = New-Object System.Windows.Media.MediaPlayer
$player.Open([System.Uri]'{uri}')
$deadline = (Get-Date).AddSeconds(5)
while (-not $player.NaturalDuration.HasTimeSpan -and (Get-Date) -lt $deadline) {{
    Start-Sleep -Milliseconds 50
}}
$player.Play()
if ($player.NaturalDuration.HasTimeSpan) {{
    Start-Sleep -Milliseconds ([Math]::Ceiling($player.NaturalDuration.TimeSpan.TotalMilliseconds) + 250)
}} else {{
    Start-Sleep -Seconds 4
}}
$player.Close()
""".strip()
    startupinfo = None
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        check=True,
        timeout=30,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo,
    )
