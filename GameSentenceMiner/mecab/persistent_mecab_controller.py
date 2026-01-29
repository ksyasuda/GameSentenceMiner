# Copyright: Ren Tatsumoto <tatsu at autistici.org> and contributors
# Copyright: Kyle Yasuda <ksyasuda at umich.edu>
# License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html
# Based on GameSentenceMiner/mecab/basic_mecab_controller.py.

import atexit
import os
import subprocess

import threading
from typing import Optional

from GameSentenceMiner.util.logging_config import logger

try:
    from .mecab_exe_finder import SUPPORT_DIR, find_executable
    from .basic_mecab_controller import (
        INPUT_BUFFER_SIZE,
        MECAB_RC_PATH,
        startup_info,
        find_best_dic_dir,
        normalize_for_platform,
        check_mecab_rc,
        expr_to_bytes,
        prepend_library_path,
    )
except ImportError:
    from mecab_exe_finder import SUPPORT_DIR, find_executable
    from basic_mecab_controller import (
        INPUT_BUFFER_SIZE,
        MECAB_RC_PATH,
        startup_info,
        find_best_dic_dir,
        normalize_for_platform,
        check_mecab_rc,
        expr_to_bytes,
        prepend_library_path,
    )


def _drain_stderr(stderr_stream) -> None:
    """Read stderr lines and log them via loguru. Runs in a daemon thread."""
    try:
        for line in stderr_stream:
            decoded = line.decode("utf-8", "replace").rstrip("\r\n")
            if decoded:
                logger.debug("MeCab stderr: {}", decoded)
    except (ValueError, OSError):
        pass


class PersistentMecabController:
    """
    A persistent MeCab controller that keeps the MeCab subprocess alive between calls.
    This significantly improves performance by avoiding the subprocess spawn overhead
    (~50-100ms on Windows) for each tokenization request.

    Thread-safe via a threading.Lock.
    Auto-recovers if the MeCab process crashes.
    """

    _DEFAULT_EOS_MARKER = "EOS"

    @staticmethod
    def _build_default_mecab_cmd() -> list[str]:
        """Build the default MeCab command. Called at instantiation time."""
        return [
            find_executable("mecab"),
            "--dicdir=" + find_best_dic_dir(),
            "--rcfile=" + MECAB_RC_PATH,
            "--userdic=" + os.path.join(SUPPORT_DIR, "user_dic.dic"),
            "--input-buffer-size=" + INPUT_BUFFER_SIZE,
        ]

    def __init__(
        self,
        mecab_cmd: Optional[list[str]] = None,
        mecab_args: Optional[list[str]] = None,
        verbose: bool = False,
    ) -> None:
        check_mecab_rc()
        self._verbose = verbose

        # Extract EOS marker from args and ensure it has a trailing newline
        # This is needed because:
        # 1. MecabController uses custom --eos-format (e.g., "<ajt__footer>")
        # 2. readline() needs a trailing newline to not block forever
        self._eos_marker = self._DEFAULT_EOS_MARKER
        modified_args = list(mecab_args) if mecab_args else []
        for i, arg in enumerate(modified_args):
            if arg.startswith("--eos-format="):
                self._eos_marker = arg.split("=", 1)[1]
                # Ensure the eos-format ends with newline for readline() to work
                if not self._eos_marker.endswith("\n"):
                    modified_args[i] = arg + "\n"
                break

        base_cmd = (
            mecab_cmd if mecab_cmd is not None else self._build_default_mecab_cmd()
        )
        self._mecab_cmd = normalize_for_platform(base_cmd + modified_args)
        prepend_library_path()

        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

        if self._verbose:
            logger.debug("mecab cmd: {}", self._mecab_cmd)
            logger.debug("eos marker: {}", repr(self._eos_marker))


    def _start_process(self) -> None:
        """Start the MeCab subprocess."""
        if self._verbose:
            logger.debug("Starting persistent MeCab process...")

        try:
            self._process = subprocess.Popen(
                self._mecab_cmd,
                bufsize=0,  # Unbuffered for immediate I/O
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startup_info(),
            )
        except OSError:
            raise Exception(
                "Please ensure your Linux system has 64 bit binary support."
            )

        # Drain stderr in a background daemon thread to prevent pipe deadlock.
        t = threading.Thread(
            target=_drain_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        t.start()

        if self._verbose:
            logger.debug(f"MeCab process started with PID {self._process.pid}")

    def _ensure_process(self) -> None:
        """Ensure the MeCab process is running, starting it if necessary."""
        if self._process is None or self._process.poll() is not None:
            if self._process is not None and self._verbose:
                logger.debug(
                    f"MeCab process terminated (code {self._process.returncode}), restarting..."
                )
            self._start_process()

    def _read_until_eos(self) -> str:
        """Read MeCab output until we see the EOS marker."""
        lines = []
        while True:
            line = self._process.stdout.readline()
            if not line:
                # Process terminated unexpectedly
                raise RuntimeError("MeCab process terminated unexpectedly")

            decoded = line.rstrip(b"\r\n").decode("utf-8", "replace")

            # With custom --node-format, the EOS marker may be on the same line as tokens
            if decoded == self._eos_marker:
                break
            elif decoded.endswith(self._eos_marker):
                content = decoded[: -len(self._eos_marker)]
                if content:
                    lines.append(content)
                break

            lines.append(decoded)

        return "\n".join(lines)

    def run(self, expr: str) -> str:
        """
        Tokenize an expression using the persistent MeCab process.
        Thread-safe and auto-recovers on process failure.

        Args:
            expr: The Japanese text to tokenize

        Returns:
            MeCab output as a string
        """
        with self._lock:
            try:
                self._ensure_process()

                input_bytes = expr_to_bytes(expr)
                self._process.stdin.write(input_bytes)
                self._process.stdin.flush()

                result = self._read_until_eos()

                if "tagger.cpp" in result and "no such file or directory" in result:
                    raise RuntimeError(
                        "Please ensure your Windows user name contains only English characters."
                    )

                return result

            except Exception as e:
                if self._process is not None:
                    try:
                        self._process.kill()
                        self._process.wait(timeout=1)
                    except Exception:
                        pass
                    self._process = None
                raise

    def is_alive(self) -> bool:
        """Check if the MeCab process is currently running."""
        with self._lock:
            return self._process is not None and self._process.poll() is None


# Global singleton for persistent MeCab controller (lazy initialization)
_persistent_mecab: Optional[PersistentMecabController] = None
_persistent_mecab_lock = threading.Lock()


def get_persistent_mecab(
    mecab_cmd: Optional[list[str]] = None,
    mecab_args: Optional[list[str]] = None,
    verbose: bool = False,
) -> PersistentMecabController:
    """
    Get the global persistent MeCab controller singleton.
    Thread-safe lazy initialization.
    """
    global _persistent_mecab

    if _persistent_mecab is None:
        with _persistent_mecab_lock:
            if _persistent_mecab is None:
                _persistent_mecab = PersistentMecabController(
                    mecab_cmd=mecab_cmd,
                    mecab_args=mecab_args,
                    verbose=verbose,
                )

    return _persistent_mecab
