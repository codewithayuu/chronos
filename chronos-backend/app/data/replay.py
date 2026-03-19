"""
Data Replay Service.

Streams pre-generated patient data to pipeline at configurable speed.
Acts as a simulated bedside monitor feed.
"""

import asyncio
from typing import List, Optional
from datetime import timedelta

from ..models import DrugEffect
from ..config import AppConfig
from .generator import PatientCase, DataGenerator

class ReplayService:
    """
    Asynchronous data replay service.

    Feeds vital-sign records from pre-generated patient cases
    into processing pipeline at a configurable speed multiplier.
    """

    def __init__(self, pipeline, config: Optional[AppConfig] = None):
        """
        Parameters
        ----------
        pipeline : ChronosPipeline
            The central processing pipeline to feed data into.
        config : AppConfig, optional
        """
        if config is None:
            config = AppConfig()
        self.pipeline = pipeline
        self.config = config.data_replay
        self._running = False
        self._current_minute = 0
        self._cases: List[PatientCase] = []
        self._max_minutes = 0

    def load_cases(self, cases: Optional[List[PatientCase]] = None):
        """Load patient cases for replay. Generates demo dataset if none provided."""
        if cases is None:
            cases = DataGenerator.generate_demo_dataset(
                num_filler=self.config.num_filler_patients,
            )
        self._cases = cases
        self._max_minutes = max(c.duration_minutes for c in cases) if cases else 0
        self._current_minute = 0
        print(f"[Replay] Loaded {len(cases)} cases, max duration: {self._max_minutes} min")

    async def run(self):
        """
        Main replay loop. Runs until stopped or cases exhausted.

        Each iteration:
          1. For each patient, emit the current minute's vital record
          2. Check for drug events at current minute
          3. Sleep for (60 / speed_multiplier) seconds
          4. Advance to next minute
        """
        self._running = True

        if not self._cases:
            self.load_cases()

        interval = 60.0 / self.config.speed_multiplier
        print(f"[Replay] Starting at {self.config.speed_multiplier}x speed "
              f"(1 record every {interval:.2f}s)")

        while self._running:
            self._tick_sync()

            self._current_minute += 1

            # Check if we've finished all cases
            if self._current_minute >= self._max_minutes:
                if self.config.loop:
                    print(f"[Replay] Looping — restarting from minute 0")
                    self._current_minute = 0
                    # Clear patient state so entropy recalibrates
                    for case in self._cases:
                        self.pipeline.remove_patient(case.patient_id)
                else:
                    print(f"[Replay] All cases complete. Stopping.")
                    self._running = False
                    break

            await asyncio.sleep(interval)

    def _tick_sync(self):
        """Process one minute of data for all patients. Synchronous."""
        minute = self._current_minute

        for case in self._cases:
            if minute >= len(case.records):
                continue

            # Emit vital record
            record = case.records[minute]
            self.pipeline.process_vital(record)

            # Check for drug events at this minute
            for drug_event in case.drug_events:
                if drug_event.minute == minute:
                    drug_effect = DrugEffect(
                        drug_name=drug_event.drug_name,
                        drug_class=drug_event.drug_class,
                        dose=drug_event.dose,
                        unit=drug_event.unit,
                        start_time=record.timestamp,
                    )
                    self.pipeline.add_drug(case.patient_id, drug_effect)
                    print(f"[Replay] 💊 {drug_event.drug_name} administered to {case.patient_id} "
                          f"at minute {minute}")

    def tick(self):
        """Public synchronous tick — for testing."""
        self._tick_sync()
        self._current_minute += 1

    def stop(self):
        """Stop the replay service."""
        self._running = False

    @property
    def current_minute(self) -> int:
        return self._current_minute

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def progress(self) -> float:
        """Replay progress as fraction (0.0 to 1.0)."""
        if self._max_minutes == 0:
            return 0.0
        return min(1.0, self._current_minute / self._max_minutes)
