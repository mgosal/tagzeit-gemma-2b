/**
 * Temporal Computation Engine — Stage 3 (Post-LLM)
 * ==================================================
 * Option B Architecture: The LLM routes, this engine computes.
 *
 * Receives structured routing calls from the LLM's output
 * and evaluates them deterministically using Luxon.
 *
 * Supported operations:
 *   ROUTE_TIME_ADD        — Add duration to a time
 *   ROUTE_TIME_SUB        — Subtract duration from a time
 *   ROUTE_DURATION_BETWEEN — Compute time difference between two references
 *   ROUTE_CALENDAR_SHIFT   — Calendar-aware date manipulation
 *   ROUTE_TIMEZONE_CONVERT — Timezone conversion
 *
 * See: Issue #17 (Arithmetic vs Calendar mode)
 */

const { DateTime, Duration, Interval } = require('luxon');

// ---------------------------------------------------------------------------
// Token Parsing Utilities
// ---------------------------------------------------------------------------

/**
 * Parse a typed token array into a structured time object.
 * e.g. ["[HEAD_TIME]", "[ARG_HOUR_14]", "[ARG_MIN_20]"] -> { hour: 14, minute: 20 }
 */
function parseTimeTokens(tokens) {
  let hour = null;
  let minute = null;

  for (const token of tokens) {
    const hourMatch = token.match(/\[ARG_HOUR_(\d{2})\]/);
    if (hourMatch) hour = parseInt(hourMatch[1], 10);

    const minMatch = token.match(/\[ARG_MIN_(\d{2})\]/);
    if (minMatch) minute = parseInt(minMatch[1], 10);
  }

  if (hour === null || minute === null) {
    throw new Error(`Cannot parse time from tokens: ${JSON.stringify(tokens)}`);
  }

  return { hour, minute };
}

/**
 * Parse a typed token array into a duration object.
 * e.g. ["[HEAD_DURATION]", "[ARG_HOUR_01]", "[ARG_MIN_30]"] -> { hours: 1, minutes: 30 }
 */
function parseDurationTokens(tokens) {
  let hours = 0;
  let minutes = 0;

  for (const token of tokens) {
    const hourMatch = token.match(/\[ARG_HOUR_(\d{2})\]/);
    if (hourMatch) hours = parseInt(hourMatch[1], 10);

    const minMatch = token.match(/\[ARG_MIN_(\d{2})\]/);
    if (minMatch) minutes = parseInt(minMatch[1], 10);
  }

  if (!hours && !minutes) {
    throw new Error(`Cannot parse duration from tokens: ${JSON.stringify(tokens)}`);
  }

  return { hours, minutes };
}

/**
 * Format a Luxon DateTime back into typed token array.
 */
function toTimeTokens(dt) {
  return [
    '[HEAD_TIME]',
    `[ARG_HOUR_${String(dt.hour).padStart(2, '0')}]`,
    `[ARG_MIN_${String(dt.minute).padStart(2, '0')}]`,
  ];
}

/**
 * Format a time result as a human-readable string.
 */
function toTimeString(dt) {
  return `${String(dt.hour).padStart(2, '0')}:${String(dt.minute).padStart(2, '0')}`;
}


// ---------------------------------------------------------------------------
// Routing Call Parser
// ---------------------------------------------------------------------------

/**
 * Parse a raw LLM output string into a structured routing call.
 *
 * Expected format from LLM:
 *   [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_58] [HEAD_DURATION] [ARG_MIN_05]
 *   [ROUTE_DURATION_BETWEEN] [REF_1] [REF_2]
 *
 * Returns: { operation, operands[] }
 */
function parseRoutingCall(raw) {
  // Extract all tokens
  const tokenRegex = /\[([A-Z_0-9]+)\]/g;
  const allTokens = [];
  let match;
  while ((match = tokenRegex.exec(raw)) !== null) {
    allTokens.push(`[${match[1]}]`);
  }

  if (allTokens.length === 0) {
    return null;
  }

  // First token should be the ROUTE operation
  const operation = allTokens[0];
  const remaining = allTokens.slice(1);

  // Group remaining tokens into operands by HEAD or REF markers
  const operands = [];
  let current = [];

  for (const token of remaining) {
    if ((token.startsWith('[HEAD_') || token.startsWith('[REF_')) && current.length > 0) {
      operands.push(current);
      current = [token];
    } else {
      current.push(token);
    }
  }
  if (current.length > 0) {
    operands.push(current);
  }

  return { operation, operands };
}


// ---------------------------------------------------------------------------
// Computation Operations (Arithmetic Mode)
// ---------------------------------------------------------------------------

/**
 * ROUTE_TIME_ADD: Add a duration to a time.
 * Handles minute carries, hour rollovers, and day boundaries.
 */
function timeAdd(timeTokens, durationTokens) {
  const time = parseTimeTokens(timeTokens);
  const duration = parseDurationTokens(durationTokens);

  const dt = DateTime.fromObject({ hour: time.hour, minute: time.minute });
  const result = dt.plus({ hours: duration.hours, minutes: duration.minutes });

  return {
    resultTokens: toTimeTokens(result),
    resultString: toTimeString(result),
    dayOverflow: result.day !== dt.day,
  };
}

/**
 * ROUTE_TIME_SUB: Subtract a duration from a time.
 * Handles borrow logic and day underflow.
 */
function timeSub(timeTokens, durationTokens) {
  const time = parseTimeTokens(timeTokens);
  const duration = parseDurationTokens(durationTokens);

  const dt = DateTime.fromObject({ hour: time.hour, minute: time.minute });
  const result = dt.minus({ hours: duration.hours, minutes: duration.minutes });

  return {
    resultTokens: toTimeTokens(result),
    resultString: toTimeString(result),
    dayUnderflow: result.day !== dt.day,
  };
}

/**
 * ROUTE_DURATION_BETWEEN: Compute the duration between two times.
 */
function durationBetween(timeTokens1, timeTokens2) {
  const t1 = parseTimeTokens(timeTokens1);
  const t2 = parseTimeTokens(timeTokens2);

  const dt1 = DateTime.fromObject({ hour: t1.hour, minute: t1.minute });
  let dt2 = DateTime.fromObject({ hour: t2.hour, minute: t2.minute });

  // If t2 < t1, assume t2 is the next day
  if (dt2 < dt1) {
    dt2 = dt2.plus({ days: 1 });
  }

  const diff = dt2.diff(dt1, ['hours', 'minutes']);

  return {
    hours: Math.floor(diff.hours),
    minutes: Math.floor(diff.minutes),
    totalMinutes: Math.floor(diff.as('minutes')),
    resultString: `${Math.floor(diff.hours)}h ${Math.floor(diff.minutes)}m`,
  };
}


// ---------------------------------------------------------------------------
// Computation Operations (Calendar Mode)
// ---------------------------------------------------------------------------

/**
 * ROUTE_CALENDAR_SHIFT: Calendar-aware date manipulation.
 * Handles weeks→months boundaries, DST transitions, leap years.
 *
 * @param {Object} anchor - { year, month, day } or a Luxon DateTime
 * @param {Object} shift  - { weeks?, months?, days?, hours?, minutes? }
 * @param {string} zone   - IANA timezone (default: 'UTC')
 */
function calendarShift(anchor, shift, zone = 'UTC') {
  let dt;

  if (anchor instanceof DateTime) {
    dt = anchor;
  } else {
    dt = DateTime.fromObject(anchor, { zone });
  }

  const result = dt.plus(shift);

  return {
    resultISO: result.toISO(),
    resultDate: result.toISODate(),
    resultTime: toTimeString(result),
    dayOfWeek: result.weekdayLong,
    dstTransition: dt.isInDST !== result.isInDST,
  };
}

/**
 * ROUTE_TIMEZONE_CONVERT: Convert a time from one timezone to another.
 */
function timezoneConvert(timeTokens, fromZone, toZone) {
  const time = parseTimeTokens(timeTokens);

  const dt = DateTime.fromObject(
    { hour: time.hour, minute: time.minute },
    { zone: fromZone }
  );
  const converted = dt.setZone(toZone);

  return {
    resultTokens: toTimeTokens(converted),
    resultString: toTimeString(converted),
    fromOffset: dt.toFormat('ZZ'),
    toOffset: converted.toFormat('ZZ'),
  };
}


// ---------------------------------------------------------------------------
// Main Dispatch — Routes a parsed call to the right operation
// ---------------------------------------------------------------------------

/**
 * Execute a routing call.
 *
 * @param {Object} routingCall - Output of parseRoutingCall()
 * @param {Object} context     - Additional context (reference map, timezone, etc.)
 * @returns {Object} Computation result
 */
function execute(routingCall, context = {}) {
  if (!routingCall || !routingCall.operation) {
    throw new Error('Invalid routing call: no operation specified');
  }

  const { operation, operands } = routingCall;

  // Universal REF resolution (Issue #7)
  const refs = context.references || {};
  const resolvedOperands = operands.map(op => {
    if (op.length === 1 && op[0].startsWith('[REF_')) {
      const refKey = op[0];
      if (!refs[refKey]) throw new Error(`Unresolved reference: ${refKey}`);
      return refs[refKey];
    }
    return op;
  });

  switch (operation) {
    case '[ROUTE_TIME_ADD]': {
      if (resolvedOperands.length < 2) throw new Error('TIME_ADD requires 2 operands (time + duration)');
      return timeAdd(resolvedOperands[0], resolvedOperands[1]);
    }

    case '[ROUTE_TIME_SUB]': {
      if (resolvedOperands.length < 2) throw new Error('TIME_SUB requires 2 operands (time + duration)');
      return timeSub(resolvedOperands[0], resolvedOperands[1]);
    }

    case '[ROUTE_DURATION_BETWEEN]': {
      if (resolvedOperands.length < 2) throw new Error('DURATION_BETWEEN requires 2 operands');
      return durationBetween(resolvedOperands[0], resolvedOperands[1]);
    }

    case '[ROUTE_CALENDAR_SHIFT]':
    case '[ROUTE_TIMEZONE_CONVERT]':
      throw new Error(`${operation} is not implemented in v1. Calendar and timezone operations are v2 scope.`);

    default:
      throw new Error(`Unknown routing operation: ${operation}`);
  }
}


// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  // Parsing
  parseTimeTokens,
  parseDurationTokens,
  parseRoutingCall,
  toTimeTokens,
  toTimeString,

  // Operations
  timeAdd,
  timeSub,
  durationBetween,
  calendarShift,
  timezoneConvert,

  // Dispatch
  execute,
};
