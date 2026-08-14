(function attachRapState(root) {
  "use strict";

  function reduceEventCache(events, event, limit) {
    const maximum = Number.isInteger(limit) && limit > 0 ? limit : 240;
    const prior = Array.isArray(events) ? events : [];
    if (!event || typeof event !== "object") return prior.slice(-maximum);

    const next = event.event_type === "session_reset" ? [] : prior.slice();
    const sequence = event.sequence;
    const duplicate = sequence !== null && sequence !== undefined
      && next.some((item) => item && item.sequence === sequence);
    if (!duplicate) next.push(event);
    return next.slice(-maximum);
  }

  root.StreamMuseRapState = Object.freeze({ reduceEventCache });
}(typeof window === "undefined" ? globalThis : window));
