"use client";

import { useEffect, useState } from "react";

/**
 * Returns a live elapsed-ms counter that ticks every second while
 * `startedAt` is truthy. Returns `undefined` when inactive.
 */
export function useElapsed(startedAt: number | undefined): number | undefined {
  const [elapsed, setElapsed] = useState<number | undefined>(() =>
    startedAt ? Date.now() - startedAt : undefined
  );

  useEffect(() => {
    if (!startedAt) return;
    const update = () => setElapsed(Date.now() - startedAt);
    const timeout = setTimeout(update, 0);
    const interval = setInterval(update, 1000);
    return () => {
      clearTimeout(timeout);
      clearInterval(interval);
    };
  }, [startedAt]);

  return startedAt ? elapsed : undefined;
}
